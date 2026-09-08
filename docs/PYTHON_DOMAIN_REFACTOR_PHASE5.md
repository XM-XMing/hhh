# Python Domain Refactor Phase 5

## Scope

This phase removes `planning.cli` and migrates Python code into domain packages while preserving the Phase 4 C++ bridge, C++ collision backend, immutable `VoxelMap`, and C++ global A* implementation.

## Package layout

```text
python/planning/
├── awac/
├── bc/
├── common/
├── contracts/
├── data/
├── diagnostics/
├── evaluation/
├── mission/
├── primitives/
├── protocol/
├── rl/
├── runtime/
├── safety/
├── teacher/
├── __init__.py
└── version.py
```

`rl/` is a temporary migration area for SAC/P3 historical code. Formal BC and AWAC remain top-level packages.

## Formal pipeline mapping

```text
scripts/generate_motion_primitives.py -> planning.primitives.generator
scripts/generate_missions.py          -> planning.mission.generator
scripts/audit_teacher_missions.py     -> planning.mission.auditor
scripts/collect_teacher_rollouts.py   -> planning.teacher.rollout_collector
scripts/label_teacher_rollouts.py     -> planning.teacher.labeling
scripts/audit_teacher_dataset.py      -> planning.teacher.dataset_audit
scripts/generate_depth_action_masks.py-> planning.safety.depth_action_masks
scripts/build_bc_mmap_dataset.py      -> planning.data.bc_mmap
scripts/train_soft_bc.py              -> planning.bc.trainer
scripts/evaluate_policy_unity.py      -> planning.evaluation.policy_evaluator
```

## Preserved Phase 4 native behavior

- Project CUDA collision source remains removed.
- Production collision backend remains C++17/OpenMP and fail-closed.
- Production global A* remains C++17 through `planning.mission.global_route`.
- Mission Audit collision backend remains CPU-only.
- Teacher relabel collision backend remains CPU-only.
- C++ `src/`, `include/`, and `CMakeLists.txt` are byte-identical to Phase 4.

## Validation

- `planning/cli`: removed.
- Production `planning.cli` imports: 0.
- Root flat implementation modules: 0; only `__init__.py` and `version.py` remain.
- Phase 4 top-level Python classes/functions: 732.
- Phase 5 top-level Python classes/functions: 732.
- Missing definitions: 0.
- `python`, `scripts`, and `tests` compileall: pass.
- Python structure/native architecture focused tests: 8/8 pass.
- Core mission/collision/route/BC/AWAC contract selection: 14/14 pass.
- Formal non-ROS pipeline script `--help`: pass for generation, audit, labeling, dataset audit, depth masks, BC mmap, and BC training.
- `collect_teacher_rollouts.py` requires `rospy`, unavailable in the current container.
- Full suite remains additionally blocked by local Unity/Mono paths and the pre-existing historical `_v4/_v1` naming contract failure; those are outside this phase.
