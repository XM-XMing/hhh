# PY7-B0 cross-language rename manifest

Date: 2026-08-29
Status: `HISTORICAL_SUPERSEDED_BY_AWAC_ONLY_MIGRATION`; no production naming
rename was executed. The detailed rows retain historical pre-migration
references for audit traceability.

```text
RENAME_MANIFEST=PASS
PRODUCTION_RENAME_EXECUTED=0
VERSIONED_NAME_OCCURRENCE_TOTAL=1033
FILE_TEXT_OCCURRENCE_TOTAL=996
PATH_SURFACE_OCCURRENCE_TOTAL=37
VERSIONED_NAME_UNIQUE_TARGET_TOTAL=319
UNIQUE_COUNT_CLOSED=YES
OCCURRENCE_COUNT_CLOSED=YES
```

The previous 283/312 labels mixed a version-marked subset with the complete
classified inventory. They are historical labels only. The current manifest
uses separate occurrence and unique-target metrics.

## Machine-readable manifest

The authoritative row-level file is
[`cross_language_rename_manifest.json`](cross_language_rename_manifest.json).
It uses the unique key `(scope, relative path, identifier)` and exactly one
of these mutually exclusive categories:

```text
KEEP_PROTOCOL_VERSION=86
KEEP_SCHEMA_VERSION=8
KEEP_ARTIFACT_VERSION=8
RENAME_PRE_BC_BUSINESS_NAME=26
DEFER_RL_NAME=50
TEST_FIXTURE_NAME=140
HISTORICAL_DOC_REFERENCE=1
UNKNOWN=0
```

```text
86+8+8+26+50+140+1+0=319
```

Every row contains `target_id`, `language`, `kind`, `old_name`, `new_name`,
`category`, `paths`, `symbols`, `occurrence_count`,
`content_occurrence_count`, `path_occurrence_count`, `reason`, `replacement`,
`public_api`, `wire_schema_impact`, `artifact_schema_impact`,
`checkpoint_dependency`, and `rename_phase`.

`occurrence_count` is the exact body-plus-relative-path count. Its body and
path components are retained separately to make the accounting reproducible.
The generated manifest itself and the two human-readable inventory documents
are excluded from the scan to avoid self-counting.

## Bounded Pre-BC candidates

Only `RENAME_PRE_BC_BUSINESS_NAME` rows enter the next phase. The following
entries are grouping records for the detailed JSON rows; they are manifest
records only and do not authorize execution:

| Group | Current owner/name | Proposed neutral name | Required gates | Status |
|---|---|---|---|---|
| RN-PY-01 | `runtime/reliable_unity_env_backend.py` reliable result/backend names | neutral reliable execution names | import graph, public aliases, runtime smoke, artifact audit | manifest only |
| RN-PY-02 | `runtime/reliable_training.py` reliable builder names | neutral reliable builder names | caller scan and compatibility aliases | manifest only |
| RN-PY-03 | `teacher/rollout_collector.py` reliable CLI/internal names | neutral reliable CLI aliases | CLI help, old-flag compatibility, resume tests | manifest only |
| RN-PY-04 | `evaluation/policy_evaluator.py` reliable CLI/internal names | neutral reliable evaluation names | checkpoint/manifest compatibility and audit smoke | manifest only |
| RN-PY-05 | collection/evaluation runtime metadata names | neutral metadata aliases if schema permits | dual-read/write and hash-input audit | blocked by frozen artifact schema |
| RN-PY-06 | managed evaluation wrapper and Pre-BC docs | neutral wrapper/documentation wording | shell contract and source-manifest audit | manifest only |

No file deletion is implied. A later rename must prove import, CLI, manifest,
checkpoint-reader, source-manifest, and test compatibility before removing an
old alias.

## Explicitly preserved/deferred names

- `PROTOCOL_VERSION=4`, `schema_version=4`, real wire/schema versions, and
  protocol golden fixtures remain `KEEP_*` identities.
- C++/Python/C# protocol namespaces and types remain preserved when they name
  the real wire contract.
- All AWAC/SAC/RL names remain `DEFER_RL_NAME` with
  `rename_phase=AFTER_NEW_RELIABLE_BC_TRAINED_AND_EVALUATED`.
- The historical `PROJECT_HANDOFF.md` `tail_v3` row is
  `HISTORICAL_DOC_REFERENCE`, not a live rename candidate.
- Unity partial filenames remain `UNITY_PARTIAL_NAMING=KEEP`; Unity was not
  modified.

```text
RENAME_EXECUTED=0
PRODUCTION_CODE_CHANGED=NO
NEXT_PHASE=PY7-B EXECUTE BOUNDED PRE-BC BUSINESS RENAME
```
