# P to O cutover plan

Date: 2026-08-28

This document records the future cutover command and rollback procedure. No
command in this document was executed in PY5.1. The historical PY5 gate was
`NO` while the full Python suite had failures; the current PY5.1 gate is
recorded at the end of this document.

## Gate summary

| Gate | Result | Evidence |
|---|---|---|
| P clean/devel build | `PASS` | `/tmp/xm-py5-devel-rfsn9K`, rc 0 |
| P install build | `PASS` | `/tmp/xm-py5-install-MXchWt`, rc 0 |
| Python import from staged install | `PASS` | `planning.__file__` is under staged install |
| Python full tests | `FAIL` | 717 passed, 42 skipped, 39 failed |
| Canonical CLI help | `PASS` | 11/11 |
| Staged MPL | `PASS` | 105 actions, 25 command frames, deterministic |
| Staged Pre-BC E2E | `PASS` | 2-worker chain through eval smoke |
| Staged 12-worker infrastructure | `PASS` | 12 identities/worker reports, cleanup complete |
| O preservation manifest | `PASS` | `O_PRESERVATION_MANIFEST.md` |
| Cutover dry-run | `PASS` | 381 regular writes, 0 deletes, 120179 data/history files preserved |
| Rollback plan | `PASS` | archive, manifest, and code-tree restore commands below |
| Final P to O readiness | `NO` | full-suite blocker remains |

## Historical assessment supersession

The older `OPTIMIZED_TREE_ASSESSMENT` / pre-migration conclusions are not
current status. They are retained as historical evidence only:

```text
OPTIMIZED_TREE_ASSESSMENT=HISTORICAL_PRE_MIGRATION_BASELINE
STATUS=SUPERSEDED_BY_CURRENT_GATES
```

In particular, the old conclusions that native route was rejected,
CUDA/hybrid removal was rejected, Bridge was static-only, and reliable
collection, BC mmap provenance, BC trainer provenance, evaluation provenance,
and relabel/mask provenance were missing are invalid as current claims. The
current PY0/PY2/PY3/PY4-A/PY5 evidence index supersedes them. The old report
is not deleted and O is not edited.

## Pre-cutover backup procedure (future, not executed)

The backup must be taken before any O code/config/document replacement. It
does not require copying the large data history archive because the cutover
rule leaves those paths untouched.

```bash
set -euo pipefail
O=/home/xm/XM/xm_ws/src/planning
BACKUP_ROOT=/home/xm/XM/archives/planning-o-pre-py5-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$BACKUP_ROOT/tree"
rsync -a --delete \
  --exclude='/data/***' \
  --exclude='/data_back_20260826/***' \
  --exclude='/build/***' \
  --exclude='/devel/***' \
  --exclude='/install/***' \
  --exclude='/.pytest_cache/***' \
  --exclude='/.benchmarks/***' \
  --exclude='*.pyc' \
  --exclude='__pycache__/***' \
  "$O/" "$BACKUP_ROOT/tree/"
tar --xattrs --acls -czf "$BACKUP_ROOT/o-code-config-tests-docs.tgz" \
  -C "$BACKUP_ROOT/tree" .
find "$BACKUP_ROOT/tree" -type f -printf '%P\t%s\n' | LC_ALL=C sort \
  | sha256sum > "$BACKUP_ROOT/tree.manifest.sha256"
sha256sum "$BACKUP_ROOT/o-code-config-tests-docs.tgz" \
  > "$BACKUP_ROOT/archive.sha256"
printf 'BACKUP_ROOT=%s\n' "$BACKUP_ROOT"
```

The backup directory is the explicit rollback path. The archive and its
sorted path/size manifest must be retained together. This procedure is a
plan only; no archive or directory rename was performed in PY5.

## Future cutover command (dry-run passed; do not run until ready)

```bash
set -euo pipefail
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
P=/home/xm/XM/src
O=/home/xm/XM/xm_ws/src/planning
rsync -a --itemize-changes \
  --exclude='/data/***' \
  --exclude='/data_back_20260826/***' \
  --exclude='/build/***' \
  --exclude='/devel/***' \
  --exclude='/install/***' \
  --exclude='/.pytest_cache/***' \
  --exclude='/.benchmarks/***' \
  --exclude='*.pyc' \
  --exclude='__pycache__/***' \
  "$P/" "$O/"

# Generate the formal MPL in O only after the code cutover and a fresh build.
export ROS_PACKAGE_PATH="$O:/opt/ros/noetic/share"
PYTHONPATH="$O/python" \
  python3 "$O/scripts/generate_motion_primitives.py" \
  --config "$O/config/motion_primitives.yaml"
```

This command has no `--delete`; therefore it cannot delete stale O code and
requires a separately reviewed cleanup decision if one is ever needed. It
cannot write O data/history through the sync. The formal generator writes
only the new MPL pair under O `data/motion_primitives/` after the gate.

## Rollback command (future, not executed)

Use the unmodified code-tree snapshot, not the staged rehearsal:

```bash
set -euo pipefail
O=/home/xm/XM/xm_ws/src/planning
BACKUP_ROOT=/home/xm/XM/archives/planning-o-pre-py5-<UTC-STAMP>
rsync -a --delete \
  --exclude='/data/***' \
  --exclude='/data_back_20260826/***' \
  --exclude='/build/***' \
  --exclude='/devel/***' \
  --exclude='/install/***' \
  --exclude='/.pytest_cache/***' \
  --exclude='/.benchmarks/***' \
  --exclude='*.pyc' \
  --exclude='__pycache__/***' \
  "$BACKUP_ROOT/tree/" "$O/"
sha256sum -c "$BACKUP_ROOT/archive.sha256"
```

The excludes protect O data and history even during rollback. A failed
post-cutover build must leave the current O tree available for inspection;
the rollback is not a reason to remove O data.

## Readiness decision

```text
P_TO_O_CUTOVER_READY=NO
NEXT_ACTION=PY5 FIX CUTOVER BLOCKERS
```

## PY5.1 current readiness gate

The old `OPTIMIZED_TREE_ASSESSMENT` remains
`HISTORICAL_PRE_MIGRATION_BASELINE`; its obsolete claims are superseded by
the later C7, M0.1, M0.2, M0.3, PY0, PY2, PY3, PY4-A, and PY5 evidence. The
full matrix is in `docs/PY5_FULL_TEST_CONTRACT_CLOSURE.md`.

```text
P_CLEAN_BUILD=PASS
P_INSTALL_BUILD=PASS
FULL_TEST_FAILURE_COUNT=0
CANONICAL_CLI=11/11
STAGED_MPL_READY=YES
STAGED_PRE_BC_E2E=PASS
STAGED_TWELVE_WORKER_SMOKE=PASS
O_PRESERVATION_MANIFEST=PASS
CUTOVER_DRY_RUN=PASS
CUTOVER_REPLACE_COUNT=396
CUTOVER_DELETE_COUNT=0
O_DATA_DELETION_COUNT=0
O_HISTORY_DELETION_COUNT=0
ROLLBACK_PLAN=PASS
ALGORITHM_CHANGED=NO
RUNTIME_BEHAVIOR_CHANGED=NO
P_TO_O_CUTOVER_READY=YES
NEXT_ACTION=PY6 EXECUTE P TO O CUTOVER
```

This is a readiness decision only. The future backup, replacement, formal
MPL generation, and rollback commands above remain unexecuted.

The 39 full-suite failures must be resolved or explicitly waived by a future
scope decision. PY5 does not modify Unity, C++, frozen AWAC/SAC behavior, or
the O tree to clear them.

## PY6 execution record

The readiness procedure above was subsequently executed under explicit PY6
authorization. The current result is recorded in
`docs/PY6_CUTOVER_REPORT.md` and is no longer a future plan:

```text
PY6_P_TO_O_CUTOVER=PASS
BACKUP=PASS
MIGRATION_REPLACE_COUNT=396
MIGRATION_ARCHIVE_COUNT=119
O_DATA_DELETION_COUNT=0
O_HISTORY_DELETION_COUNT=0
O_PRESERVED_FILE_MUTATION_COUNT=0
O_CLEAN_BUILD=PASS
O_DEVEL_BUILD=PASS
O_INSTALL_BUILD=PASS
O_PYTHON_IMPORT=PASS
FULL_PYTHON_TESTS=756 passed, 42 skipped, 0 failed
CANONICAL_CLI=11/11
FORMAL_MPL=PASS
O_PRE_BC_E2E=PASS
O_TWELVE_WORKER_SMOKE=PASS
RUNTIME_IDENTITY=PASS
PROCESS_CLEANUP=PASS
ROLLBACK_EXECUTED=NO
ALGORITHM_CHANGED=NO
RUNTIME_BEHAVIOR_CHANGED=NO
NEXT_ACTION=USER EXECUTES FORMAL RELIABLE BC DATA COLLECTION
```

The procedure preserved `data/` and `data_back_20260826/`; only the formal
MPL pair was added under `data/motion_primitives/`. No formal 60,000-row
collection, formal BC training, AWAC/RL run, Unity modification, or commit was
performed. The O clean/install Bridge binaries were not substituted for the
explicitly frozen C7 runtime Bridge used in the runtime smoke.
