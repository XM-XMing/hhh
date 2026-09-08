# PY3 pre-BC keep set and reachability inventory

> STATUS: HISTORICAL / SUPERSEDED_BY_SHELL_ENTRYPOINT_CLEANUP_AND_TEST_SUITE_INTEGRITY_V1.
> The wrapper keep-set and counts below describe the pre-AWAC-migration
> inventory; current shell ownership is recorded in the current cleanup report.

Date: 2026-08-28

The inventory covers non-generated files under:

```text
python/planning
scripts
config
launch
tests
```

Existing `__pycache__/**` and `*.pyc` files are excluded from the source
inventory. No file was deleted. The complete machine-readable classification
is `/tmp/xm-pre-bc-py3/inventory_classification.json`.

## Counts

| Classification | Count | Assignment rule |
|---|---:|---|
| `KEEP_PRE_BC` | 111 | Canonical MPL, mission, Teacher, reliable runtime, relabel, mask, dataset audit, BC, evaluation, protocol, launch, config, and their shared non-RL dependencies |
| `KEEP_RL` | 36 | Everything under `python/planning/awac`, `python/planning/rl`, plus the SAC/AWAC scripts listed below |
| `KEEP_RL_TEMPORARY_DEPENDENCY` | 3 | Shared contract modules currently required by frozen AWAC/RL checkpoint and handoff imports |
| `KEEP_TEST_REFERENCE` | 223 | All 189 tests/fixtures/benchmarks plus diagnostic/reference modules and scripts not on the formal pre-BC path |
| `DELETE_DUPLICATE` | 0 | No duplicate was proven unreachable and behavior-free |
| `DELETE_LEGACY_FORMAL` | 0 | No legacy formal owner was proven removable; legacy async behavior remains explicit diagnostic/reference only |
| `DELETE_DEAD` | 0 | No dead file was proven by import/build/test reachability |
| `DEFER_UNKNOWN` | 0 | Every inventoried file is retained in a named protected set |

Total inventoried non-generated files: `373`.

## Formal pre-BC owner set

The `KEEP_PRE_BC` set contains the canonical owners:

```text
python/planning/primitives
python/planning/mission
python/planning/teacher
python/planning/runtime
python/planning/data
python/planning/safety
python/planning/bc
python/planning/evaluation
python/planning/protocol
python/planning/native
python/planning/common
python/planning/contracts (except the three RL temporary dependencies)
config/**
launch/**
scripts/{generate_motion_primitives.py,generate_missions.py,
audit_teacher_missions.py,collect_rollouts_parallel.py,
collect_rollouts_parallel.sh,collect_teacher_rollouts.py,
merge_teacher_rollouts.py,label_teacher_rollouts.py,
audit_teacher_dataset.py,generate_depth_action_masks.py,
build_bc_mmap_dataset.py,train_soft_bc.py,evaluate_policy_unity.py,
evaluate_policy_unity_managed.sh}
```

The small shared diagnostics subset retained in `KEEP_PRE_BC` is:

```text
python/planning/diagnostics/__init__.py
python/planning/diagnostics/action_distribution.py
python/planning/diagnostics/observability.py
python/planning/diagnostics/pairing_counterfactual.py
```

These are imported by the mission audit, Teacher collection, BC trainer, or
policy evaluator and therefore are not deletion candidates.

## Frozen RL set

All of the following remain protected and were not run or changed in PY3:

```text
python/planning/awac/**
python/planning/rl/**
scripts/audit_sac_actor_replay.py
scripts/build_sac_final_holdout.py
scripts/build_sac_mission_split.py
scripts/diagnose_sac_run.py
scripts/rollback_sac_actor.py
scripts/run_awac_guarded.sh
scripts/run_awac_reliable_v4.sh
scripts/run_sac_guarded.sh
scripts/select_sac_dev_checkpoint.py
scripts/train_sac.py
```

The three temporary RL dependencies are explicitly:

```text
python/planning/contracts/awac_handoff.py
python/planning/contracts/offpolicy.py
python/planning/contracts/policy_checkpoint_fingerprint.py
```

`planning/rl`, AWAC imports, SAC modules, checkpoint/resume helpers, and
guarded runners remain in the keep set even though the next phase is pre-BC
cleanup.

## Reference and diagnostic set

All `tests/**` are `KEEP_TEST_REFERENCE`, including C++/C# protocol fixtures,
Unity integration contracts, O/P comparators, and RL tests. The remaining
reference tools include depth/base-system checks, collision debug CLI,
transport/pairing/cross-mission diagnostics, visualization, and the
diagnostic legacy single-worker path. Their presence is intentional: they are
not formal owners, but they are required to preserve regression evidence until
PY4 has an explicit deletion decision.

## Historical PY3 operational note

At PY3 time the P shell collection wrapper was retained in `KEEP_PRE_BC`, and
the smoke used its direct Python entrypoint because the wrapper could select O
after re-sourcing the wrong workspace. PY4-A closed that bootstrap issue; the
historical note is retained to explain the PY3 command record, not as a current
failure.

## PY4-A production surface audit

PY4-A closed the wrapper bootstrap seam without changing collection logic. The
wrapper now clears caller arguments while sourcing setup, recognizes source,
devel, and install layouts, and prepends the P package/ROS paths. Its source,
devel, install, arbitrary-cwd, and explicit-workspace help matrix passed. The
wrapper remains `KEEP_PRE_BC`; it is not a deletion candidate.

The production install surface is 132 Python module files in 16 packages, 24
Python scripts, 4 shell scripts, 2 configs, 8 launch files, and 3 RViz files.
Tests and docs are not installed. Diagnostics remain explicitly retained
reference/debug seams, and the frozen RL/AWAC package remains protected.

```text
PY4_A_PRODUCTION_SURFACE_AUDIT=PASS
SAFE_DELETION_BATCH=[]
```

## PY5 cleanup and cutover preservation

PY5 removed only 56 confirmed P source `.pyc` files and their cache
directories. The resulting P source tree has zero `.pyc`, zero
`__pycache__`, and no source-tree build/devel/install output. Tests, fixtures,
golden vectors, benchmark source, qualification evidence, and documentation
remain in the keep set.

The production install classification is now explicit: zero tests, zero
fixtures, four required shared diagnostic package files, 22 operator/reference
diagnostic files, and zero qualification-only installed files. The 26
diagnostic/reference files are retained; this classification is not a deletion
authorization.

The staged cutover tree and all generated rehearsal artifacts live under
`/tmp/xm-planning-cutover-stage/` and `/tmp/xm-py5-staged-e2e/`. They are not
formal P data and were not copied into O. O `data/`,
`data_back_20260826/`, map assets, checkpoints, evaluations, Teacher history,
and user-owned artifacts remain preserved by
`docs/O_PRESERVATION_MANIFEST.md`.

```text
PY5_SOURCE_GENERATED_BEFORE=56
PY5_SOURCE_GENERATED_AFTER=0
O_DATA_DELETION_COUNT=0
O_HISTORY_DELETION_COUNT=0
P_TO_O_CUTOVER_READY=NO
```
