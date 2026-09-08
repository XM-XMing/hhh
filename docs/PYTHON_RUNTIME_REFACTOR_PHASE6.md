# Python Runtime Refactor Phase 6

> STATUS: HISTORICAL / SUPERSEDED_BY_SHELL_ENTRYPOINT_CLEANUP_AND_TEST_SUITE_INTEGRITY_V1.
> The Phase 6 collection compatibility wrapper described below has since been
> removed; `scripts/collect_rollouts_parallel.py` is the current entrypoint.

## Scope

This phase keeps the Phase 5 domain layout and the Phase 4 C++ implementation unchanged.

Changes are limited to Python runtime/common ownership and the parallel teacher collection launcher.

## Runtime ownership

```text
scripts/collect_rollouts_parallel.sh
    -> scripts/collect_rollouts_parallel.py
    -> planning.teacher.parallel_collection
        -> planning.runtime.supervisor
            -> planning.runtime.collection_worker.RuntimeWorker
                -> planning.runtime.process.ManagedProcess
                -> planning.runtime.health
                -> planning.runtime.ports
        -> planning.data.rollout_merge.merge_worker_rollouts
```

## Shell reduction

`collect_rollouts_parallel.sh` changed from 556 LOC to 6 LOC. It only resolves the workspace, sources `devel/setup.bash`, and executes the Python entry point.

The shell no longer owns PID arrays, port derivation, ROS readiness checks, global progress counters, stop-file decisions, or merge orchestration.

## Common consolidation

`planning.common.io` now owns shared JSON/text/bytes atomic replacement. `planning.common.atomic` remains only as a compatibility import for temporary RL code.

`planning.common.hashing` now contains the shared file and bytes SHA-256 primitives.

## Behavior preserved

The Python launcher preserves the legacy collection contract:

- same positional arguments: `MISSION_INDEX OUT_DIR [WORKERS]`
- same environment defaults
- same worker clipping by `TARGET_ACCEPTED` and `MAX_EPISODES`
- same ROS master, command, state, depth port formulas
- same Unity command line arguments
- same `sim_realtime.launch` arguments
- same collector arguments and environment contract
- same 2-second progress polling
- same global stop-file semantics
- same final merge quality gate

## Verification

- Python compileall: PASS
- shell syntax: PASS
- runtime/domain/merge/native selected tests: 16/16 PASS
- `planning.cli` production references: 0
- compatibility shell LOC: 6
- Python entry LOC: 7
- Phase 5 vs Phase 6 C++ `include/`: 0 changed files
- Phase 5 vs Phase 6 C++ `src/`: 0 changed files
- generated Python/pytest caches removed before packaging
