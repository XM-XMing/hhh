# P to O migration manifest

Date: 2026-08-28

```text
SOURCE_P=/home/xm/XM/src
TARGET_O=/home/xm/XM/xm_ws/src/planning
O_WRITE_POLICY=READ_ONLY_DURING_PY5
```

This is a controlled replacement manifest. It is not an instruction to run
the cutover in PY5. O code and historical documents must be archived before
any future replacement; O data and historical artifacts remain in place.

## Migration matrix

| PATH | SOURCE | TARGET | ACTION |
|---|---|---|---|
| `CMakeLists.txt` | P | O | `REPLACE_WITH_P` |
| `package.xml` | P | O | `REPLACE_WITH_P` |
| `setup.py` | P | O | `REPLACE_WITH_P` |
| `pyproject.toml` | P | O | `REPLACE_WITH_P` |
| `include/` | P | O | `REPLACE_WITH_P` |
| `src/` | P | O | `REPLACE_WITH_P` |
| `python/` | P | O | `REPLACE_WITH_P` |
| `scripts/` | P | O | `REPLACE_WITH_P` |
| `config/` | P | O | `REPLACE_WITH_P` |
| `launch/` | P | O | `REPLACE_WITH_P` |
| `msg/` | P | O | `REPLACE_WITH_P` |
| `tests/` | P | O | `REPLACE_WITH_P` |
| `docs/` | P | O | `REPLACE_WITH_P` |
| `rviz/` | P | O | `REPLACE_WITH_P` |
| `data/` | O | O | `PRESERVE_O`; never sync P over this path |
| `data_back_20260826/` | O | O | `PRESERVE_O`; never sync P over this path |
| `P/data/motion_primitives/` | staged only | O `data/motion_primitives/` | `REBUILD_AFTER_CUTOVER`, using the P config and generator |
| `build/` | none | O | `REBUILD_AFTER_CUTOVER` |
| `devel/` | none | O | `REBUILD_AFTER_CUTOVER` |
| `install/` | none | O | `REBUILD_AFTER_CUTOVER` |
| `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.benchmarks/` | generated | O | `EXCLUDE_GENERATED` |

P contains 479 source regular files in the cutover source set. The staged
rehearsal contains 482 non-generated files: those 479 P files plus three
staged data files (the generated MPL pair and the required voxel cache). P
does not currently contain a formal `data/motion_primitives` pair.

Every `REPLACE_WITH_P` row requires the corresponding O tree to be archived
first. The O `docs/` row is therefore an archive-then-replace operation even
though its matrix action is the canonical `REPLACE_WITH_P` value.

## Data and history boundary

The source P sync excludes `/data/***` and `/data_back_20260826/***` from O.
The required current map asset remains the O asset. The 2026-08-26 archive,
including BC/AWAC/SAC checkpoints, evaluation evidence, Teacher rollouts,
smoke artifacts, tests, and debug artifacts, is preserved by the manifest in
`O_PRESERVATION_MANIFEST.md`.

No O build/devel/install tree is copied. A post-cutover clean catkin build is
required, followed by the formal MPL generation command. The staged MPL
hashes are qualification evidence only, not a request to copy staged data
into O during PY5.

## Frozen boundaries

The matrix does not change Unity, C++ behavior, BC/AWAC mathematics,
protocols, or the frozen runtime artifact identities. It does not merge the
three common capabilities or the thirteen duplicate configuration-owner
families. The three direct AWAC-to-SAC import edges remain protected.

```text
DUPLICATE_COMMON_CAPABILITIES_DEFERRED=3
DUPLICATE_CONFIG_OWNERS_DEFERRED=13
AWAC_TO_SAC_IMPORT_COUNT=3
ALGORITHM_CHANGED=NO
DATA_LAYOUT_CHANGED=NO
```

## Dry-run evidence

The exact source-to-target dry-run was recorded at
`/tmp/xm-py5-cutover-rsync-dry-run.txt` with `--dry-run --itemize-changes`
and explicit data/history/build/cache exclusions:

```text
CUTOVER_DRY_RUN_RC=0
CUTOVER_REPLACE_COUNT=381
CUTOVER_DELETE_COUNT=0
CUTOVER_PRESERVE_COUNT=120179
O_DATA_DELETION_COUNT=0
O_HISTORY_DELETION_COUNT=0
```

`CUTOVER_REPLACE_COUNT` counts regular-file `>f` itemizations, including new
P files and changed O files. `CUTOVER_PRESERVE_COUNT` counts the 120,179
regular files currently under O `data/` and `data_back_20260826/`; six O
generated-cache files are additionally excluded from the sync. The dry-run
performed no filesystem mutation.

## PY5.1 full test contract closure (current)

The preceding entries are historical phase records. The current controlled
migration gate is:

```text
PY5_1_FULL_TEST_CONTRACT_CLOSURE=PASS
INITIAL_TEST_FAILURE_COUNT=39
CLASSIFIED_FAILURE_COUNT=39
UNKNOWN_FAILURE_COUNT=0
FINAL_TEST_FAILURE_COUNT=0
FULL_PYTHON_TESTS=PASS
CANONICAL_CLI=11/11
SKIP_COUNT_BEFORE=42
SKIP_COUNT_AFTER=42
XFAIL_COUNT_BEFORE=0
XFAIL_COUNT_AFTER=0
```

All 39 initial failures were stale path/fixture/text contracts: 10 Unity
owner paths, four P0 fake-runtime fields, 24 guarded AWAC/SAC setup-root
fixtures, and one version-naming assertion. The full row-level record is
`docs/PY5_FULL_TEST_CONTRACT_CLOSURE.md`. The only production file changed
was `scripts/run_sac_guarded.sh`; AWAC/SAC math, cadence, checkpoint/resume,
selection, and guard semantics remain frozen.

The current P source set has 482 non-generated files, with deterministic
path/size manifest SHA256
`5b6fa89c65011606c556cc92d879173cf0ae1d13ea12804ccddaae12c0f3360d`.
The current cutover rehearsal is
`/tmp/xm-py5-1-cutover-rsync-dry-run-final.txt`:

```text
CUTOVER_DRY_RUN=PASS
CUTOVER_REPLACE_COUNT=396
CUTOVER_DELETE_COUNT=0
```

The post-production staged chain is recorded at
`/tmp/xm-py5-1-staged-e2e-final/` and passed the bounded 2-worker Pre-BC
chain, one-epoch BC smoke, audit-only reliable evaluation, and the 12-worker
infrastructure smoke. The 12-worker parent quality gate was intentionally
false because `TARGET_ACCEPTED=0`; all 12 workers completed with unique
identities/ports and zero collector or reliability errors.

The O protection check is `/tmp/xm-py5-1-o-preservation-final.txt` and
matches the existing protected inventory/content hashes:

```text
O_PRESERVATION_MANIFEST=PASS
O_DATA_DELETION_COUNT=0
O_HISTORY_DELETION_COUNT=0
O_DATA_MANIFEST_SHA256=71e21ea091be6c075651d3bb727497c530c40081a30188af2cb45422450fef7b
O_HISTORY_MANIFEST_SHA256=6b600703263bf3b7f0916fe61a92deaa869278fa99ebb16ccb654bd324e84f41
ROLLBACK_PLAN=PASS
ALGORITHM_CHANGED=NO
DATA_LAYOUT_CHANGED=NO
P_TO_O_CUTOVER_READY=YES
NEXT_PHASE=PY6 EXECUTE P TO O CUTOVER
```

No cutover, formal data collection, or O write was performed.
