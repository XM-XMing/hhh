# Shell entrypoint cleanup and test-suite integrity V1

Date: 2026-09-03  
Scope: `/home/xm/XM/xm_ws/src/planning` only  
Workspace: `/home/xm/XM/xm_ws`

## Decision

```text
SHELL_ENTRYPOINT_CLEANUP_AND_TEST_SUITE_INTEGRITY_V1=PASS
TOTAL_SOURCE_SHELL_COUNT_BEFORE=10
TOTAL_SOURCE_SHELL_COUNT_AFTER=7
RL_AWAC_SAC_PRODUCTION_SHELL_COUNT=0
GENERIC_EVALUATION_SHELL_REQUIRED=YES
```

The three deleted files were compatibility or algorithm-selection wrappers:

- `scripts/collect_rollouts_parallel.sh` only bootstrapped the Python
  collector and duplicated environment/path setup;
- `scripts/run_awac_guarded.sh` only forwarded to `train_awac.py`;
- `scripts/run_awac_reliable_v4.sh` only forwarded to `train_awac.py
  --reliable-v4`.

No shell-owned business logic was migrated in this slice because these three
files had no unique logic. The Python entrypoints already own the contracts.
`scripts/evaluate_policy_unity_managed.sh` remains required: it owns the
managed ROS master, Unity, Bridge lifecycle, readiness, signal cleanup, and
atomic evaluation publication. The six remaining shells are diagnostic
orchestration wrappers around that managed lifecycle and are not CMake
installed formal AWAC/SAC/Collection entrypoints.

## Shell inventory

The inventory was taken before deletion from the source tree, excluding
generated `build/`, `devel/`, and data roots.

| Shell | Current role | Called by / calls | Python equivalent | Decision |
|---|---|---|---|---|
| `collect_rollouts_parallel.sh` | legacy compatibility bootstrap | CMake, historical docs, pre-change tests; called `collect_rollouts_parallel.py` | `scripts/collect_rollouts_parallel.py` | `DELETE` |
| `evaluate_policy_unity_managed.sh` | managed evaluation lifecycle | CMake and diagnostic shells; starts ROS/Unity/Bridge and calls evaluator | `scripts/evaluate_policy_unity.py` plus lifecycle wrapper | `KEEP` |
| `run_awac_guarded.sh` | AWAC thin forwarding wrapper | CMake, historical docs, pre-change tests; called `train_awac.py` | `scripts/train_awac.py` | `DELETE` |
| `run_awac_reliable_v4.sh` | reliable-v4 thin forwarding wrapper | CMake, historical docs, pre-change tests; called `train_awac.py --reliable-v4` | `scripts/train_awac.py --reliable-v4` | `DELETE` |
| `run_admissible_pairing_audit.sh` | evaluation diagnostic | calls managed evaluator and parses diagnostic JSON | evaluator + diagnostic Python modules | `KEEP` |
| `run_command_tick_audit.sh` | evaluation diagnostic | repeats managed evaluator and calls tick diagnosis | evaluator + diagnostic Python modules | `KEEP` |
| `run_cross_mission_t2_baseline_audit.sh` | evaluation diagnostic | calls managed evaluator and checks artifacts | evaluator + diagnostic Python modules | `KEEP` |
| `run_cross_mission_t2_risk_screen.sh` | evaluation diagnostic | loops managed evaluator and calls risk diagnosis | evaluator + diagnostic Python modules | `KEEP` |
| `run_live_pairing_reachability.sh` | evaluation diagnostic | repeats managed evaluator and calls reachability diagnosis | evaluator + diagnostic Python modules | `KEEP` |
| `run_policy_eval_repro.sh` | evaluation diagnostic | repeats managed evaluator and calls divergence diagnosis | evaluator + diagnostic Python modules | `KEEP` |

There was no current `run_sac_*.sh` file. Historical references remain in
historical evidence documents only and are not source callers.

## CMake and active documentation

`CMakeLists.txt` now installs the Python scripts and only the required managed
evaluation shell. It no longer installs the deleted collection/AWAC shells.
`package.xml` already described the current package as BC-initialized discrete
AWAC and required no change.

Current formal entrypoints are:

```text
FORMAL_AWAC_ENTRYPOINT=scripts/train_awac.py
FORMAL_POLICY_EVAL_ENTRYPOINT=scripts/evaluate_policy_unity.py
TRAIN_AWAC_PY_ENTRYPOINT=PASS
EVALUATE_POLICY_PY_ENTRYPOINT=PASS
AUDIT_AWAC_REPLAY_PY_ENTRYPOINT=PASS
SELECT_AWAC_DEV_CHECKPOINT_PY_ENTRYPOINT=PASS
```

Each entrypoint was checked with the active `xm` conda interpreter and
`--help`; no training, replay collection, Unity evaluation, or AWAC run was
started. The current handoff document was corrected so it no longer presents
the deleted shell names as commands. Earlier phase documents retain their
original evidence and now carry an explicit historical/superseded marker.

## Test count audit

The two source logs are exact:

```text
OLD_FULL_TEST_PASS_COUNT=891
OLD_FULL_TEST_SKIP_COUNT=42
CURRENT_PRE_AUDIT_PASS_COUNT=680
CURRENT_PRE_AUDIT_SKIP_COUNT=42
NET_PASS_COUNT_DROP=211
```

The skip count did not increase. The AWAC-only migration report records that
the old SAC-only learner, candidate-funnel, rollback, guard, audit, trainer,
and script surfaces were removed after the caller/test/CMake zero-reference
gate. The current collection inventory before this cleanup was 721 collected
cases from 115 contributing test files. The cleanup adds two infrastructure
tests, so the final collection is expected to be 723 cases from 116 files;
the final full run below is the authoritative result.

The exact `211` is the observed net pass-count delta, not a count of files.
The 28-row historical mapping in the JSON report is a separate coverage
mapping. Five reusable rows have current equivalents; the 23 remaining rows
are bound to the deleted SAC/AWAC guarded-runner contract and are explicitly
`DELETED_SAC_ONLY`. They are not silently omitted. Thus:

```text
INTENTIONAL_REMOVED_TEST_CASES=211  (net observed migration delta)
MIGRATED_EQUIVALENT_TEST_CASES=5   (coverage mapping, not added to delta)
SUPERSEDED_TEST_CASES=0
ACCIDENTALLY_MISSING_TEST_CASES=0
UNEXPLAINED_TEST_COUNT=0
```

No current collection break, import-path loss, or accidental missing behavior
was found. The old guarded-runner rows are not retained as compatibility
tests because the runner and its SAC-coupled contract no longer exist.

## Reusable behavior coverage

The following current owners retain the reusable behavior required by the
audit:

| Behavior | Current tests |
|---|---|
| BC to AWAC checkpoint, strict identity, logits/input/action dimensions | `test_awac_handoff_provenance.py`, `test_awac_handoff_logits.py`, `test_awac_trainer_migration.py` |
| Actor/Critic, Twin-Q, masked policy, Bellman terminal behavior, AWAC weights and KL/CQL helpers | `test_awac_core.py`, `test_awac_model_migration.py`, `test_awac_schedule.py`, `test_awac_actor_onset.py` |
| Replay append/sample/persistence/transaction/finite fields | `test_awac_replay_migration.py`, `test_awac_replay_transaction.py`, `test_awac_replay_audit.py`, `test_awac_training_runtime_contract.py` |
| Runtime worker identity, port isolation, cleanup, terminal handling, reliable-exact lifecycle | `test_parallel_env.py`, `test_parallel_terminal_abort_contract.py`, `test_reliable_v4_runtime_instance_lifecycle.py`, `test_reliable_v4_worker_runtime_spec.py`, `test_collection_cli_logging.py` |
| Checkpoint atomicity/resume identity/contract hashes | `test_awac_checkpoint_migration.py`, `test_checkpoint_evidence.py`, `test_awac_trainer_migration.py` |
| Evaluation checkpoint/runtime contract and dev selection | `test_awac_dev_selection.py`, `test_evaluator_terminal_abort_contract.py`, `test_cli_normalization.py` |
| Reward numeric and terminal semantics | `test_reward_contract_migration.py`, task/reward contract tests |

Therefore `REUSABLE_BEHAVIOR_TEST_COVERAGE_PRESERVED=YES`. The removed tests
for SAC entropy/alpha, candidate interaction, old rollback thresholds, old
SAC output ownership, and the deleted guarded runner are intentionally not
reintroduced under an AWAC name.

## Historical 28-row mapping

The detailed machine-readable rows are in
`docs/shell_entrypoint_cleanup_and_test_suite_integrity_v1.json`. The
explicitly documented F15--F38 rows come from the historical PY5 failure
matrix; H1--H4 are the reusable owner migrations recorded by the AWAC
ownership audit.

| ID | Historical test | Current equivalent | Status |
|---|---|---|---|
| F15 | guarded runner dry-run/disjoint manifest | — | `DELETED_SAC_ONLY` |
| F16 | independent SAC training/full-eval cadence | — | `DELETED_SAC_ONLY` |
| F17 | AWAC wrapper/shared SAC runner dry-run | — | `DELETED_SAC_ONLY` |
| F18 | non-aborted checkpoint on resume | `test_awac_trainer_migration.py` | `MIGRATED` |
| F19--F27 | baseline/evaluator/collection guards | — | `DELETED_SAC_ONLY` |
| F28 | SAC output-directory refusal | — | `DELETED_SAC_ONLY` |
| F29--F33 | guarded protected-argument checks | — | `DELETED_SAC_ONLY` |
| F34 | intervention-required guarded state | — | `DELETED_SAC_ONLY` |
| F35--F36 | guarded deployable-policy identity recovery | — | `DELETED_SAC_ONLY` |
| F37--F38 | guarded catastrophic-result threshold | — | `DELETED_SAC_ONLY` |
| H1 | `test_sac_core.py` | `test_awac_core.py`, `test_awac_model_migration.py` | `MIGRATED` |
| H2 | `test_sac_replay_transaction.py` | `test_awac_replay_transaction.py` | `MIGRATED` |
| H3 | `test_sac_parallel_env.py` | `test_parallel_env.py` | `MIGRATED` |
| H4 | `test_sac_training_contract.py` | `test_awac_training_runtime_contract.py`, `test_awac_trainer_migration.py` | `MIGRATED` |

The ranges in the table expand to 28 individual rows in the JSON manifest.

## Frozen scope and verification

```text
CONFIDENCE_ALGORITHM_IMPLEMENTED=NO
PRIMITIVE_NEIGHBOR_ALGORITHM_IMPLEMENTED=NO
CRITIC_CALIBRATION_IMPLEMENTED=NO
BC_CHANGED=NO
ALGORITHM_CHANGED=NO
UNITY_CHANGED=NO
BRIDGE_CHANGED=NO
TASK_CONTRACT_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
AWAC_TO_RL_IMPORT_COUNT=0
AWAC_TO_SAC_IMPORT_COUNT=0
SAC_PRODUCTION_FILE_COUNT=0
RL_DIRECTORY_EXISTS=NO
COMMIT=NO
```

No C++ build, Unity process, collection, BC training, AWAC training, replay
collection, Critic Calibration, or final evaluation was run. The final
verification commands were run after `conda activate xm`:

```bash
python -m compileall -q python/planning scripts
python -m pytest -q
```

The machine-readable report records the exact command outcomes and final
counts. The next action remains the separately authorized RL Phase 0 AWAC
contract and Critic Calibration implementation; it was not started here.
