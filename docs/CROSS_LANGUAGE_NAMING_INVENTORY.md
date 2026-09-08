# PY7-B0 cross-language naming inventory

Date: 2026-08-29
Status: `HISTORICAL_SUPERSEDED_BY_AWAC_ONLY_MIGRATION`; no naming rename was
executed. The JSON rows retain the pre-migration path inventory for audit
traceability; current AWAC ownership is documented separately.

This document normalizes the previous `283` versus `312` count mismatch. The
two numbers measured different things and are no longer used as current
metrics.

## Scope and accounting units

- `O=/home/xm/XM/xm_ws/src/planning` is the only writable root.
- `U=/home/xm/XM/XMflight` is frozen and read-only; only its
  `Assets/Scripts` C# surface is included for the cross-language and partial
  class audit.
- The unique-target base is the audited 319-row inventory, keyed by
  `(scope, relative_path, identifier)`. Each key occurs in exactly one
  category.
- An occurrence is one exact `old_name` in a scanned file body or in that
  file's relative path surface. Body and path counts are reported separately
  so filename-only identities cannot be mistaken for body text.
- Generated inventory outputs, build/install/cache/history directories, and
  data artifacts are excluded from the scan. The machine-readable rows and
  required fields are in
  [`cross_language_rename_manifest.json`](cross_language_rename_manifest.json).

```text
VERSIONED_NAME_OCCURRENCE_TOTAL=1033
FILE_TEXT_OCCURRENCE_TOTAL=996
PATH_SURFACE_OCCURRENCE_TOTAL=37
VERSIONED_NAME_UNIQUE_TARGET_TOTAL=319
```

`VERSIONED_NAME_OCCURRENCE_TOTAL` is the sum of every row's
`occurrence_count`; `FILE_TEXT_OCCURRENCE_TOTAL` and
`PATH_SURFACE_OCCURRENCE_TOTAL` are its auditable components.

## Mutually exclusive classification

```text
KEEP_PROTOCOL_VERSION_UNIQUE_COUNT=86
KEEP_SCHEMA_VERSION_UNIQUE_COUNT=8
KEEP_ARTIFACT_VERSION_UNIQUE_COUNT=8
RENAME_PRE_BC_BUSINESS_NAME_UNIQUE_COUNT=26
DEFER_RL_NAME_UNIQUE_COUNT=50
TEST_FIXTURE_NAME_UNIQUE_COUNT=140
HISTORICAL_DOC_REFERENCE_UNIQUE_COUNT=1
UNKNOWN_UNIQUE_COUNT=0
UNIQUE_COUNT_CLOSED=YES
OCCURRENCE_COUNT_CLOSED=YES
```

The mutually exclusive sum is:

```text
86+8+8+26+50+140+1+0=319
```

| Category | Meaning | Action in PY7-B0 |
|---|---|---|
| `KEEP_PROTOCOL_VERSION` | Frozen wire/protocol identity and golden-vector names | Preserve |
| `KEEP_SCHEMA_VERSION` | Serialized schema revision | Preserve |
| `KEEP_ARTIFACT_VERSION` | Artifact, provenance, lifecycle, or manifest identity | Preserve |
| `RENAME_PRE_BC_BUSINESS_NAME` | Pre-BC business-facing version suffix | Candidate only for PY7-B |
| `DEFER_RL_NAME` | AWAC/SAC/RL or RL evidence identity | Defer until `AFTER_NEW_RELIABLE_BC_TRAINED_AND_EVALUATED` |
| `TEST_FIXTURE_NAME` | Test, fixture, or golden-vector identity | Preserve oracle identity |
| `HISTORICAL_DOC_REFERENCE` | Historical documentation reference only | Retain for audit traceability |
| `UNKNOWN` | Unclassified target | Required zero |

`tail_v3` in `PROJECT_HANDOFF.md` is the single
`HISTORICAL_DOC_REFERENCE`; it is not a live Pre-BC rename target. The 26
Pre-BC candidates are the only rows with `rename_phase=PY7-B`.

## Preserved version identities

The following remain unchanged:

- `PROTOCOL_VERSION=4` and `schema_version=4` wire/schema identities.
- Real checkpoint, dataset, artifact, bridge, and provenance schema versions.
- Protocol v4 golden fixtures and the C++/C#/Python protocol namespaces and
  types when they identify the real wire contract.
- Unity partial filenames using `MainType.Responsibility.cs`.

No protocol bytes, schema fields, checkpoint identity, fixture oracle, or
runtime behavior was changed by PY7-B0.

## Unity partial-file audit

The frozen files below each declare the same `XMflight.XMSimulationManager`
partial class exactly once:

```text
Assets/Scripts/Runtime/Simulation/XMSimulationManager.cs
Assets/Scripts/Runtime/Simulation/XMSimulationManager.CommandDecoding.cs
Assets/Scripts/Runtime/Simulation/XMSimulationManager.Diagnostics.cs
Assets/Scripts/Runtime/Simulation/XMSimulationManager.Execution.cs
Assets/Scripts/Runtime/Simulation/XMSimulationManager.MotionCommands.cs
Assets/Scripts/Runtime/Simulation/XMSimulationManager.Observation.cs
Assets/Scripts/Runtime/Simulation/XMSimulationManager.Telemetry.cs
Assets/Scripts/Runtime/Simulation/XMSimulationManager.TransportPolling.cs
```

The suffix files group orchestration glue by responsibility; no business owner
leak or naming regression was found.

```text
UNITY_PARTIAL_NAMING=KEEP
```

## Historical assessment boundary

The old `OPTIMIZED_TREE_ASSESSMENT` is retained only as
`HISTORICAL_PRE_MIGRATION_BASELINE` with status `SUPERSEDED`. Its former
next-migration recommendation is not a current instruction. The normalized
machine-readable manifest is an inventory, not an execution plan.

```text
PRODUCTION_RENAME_EXECUTED=0
PRODUCTION_CODE_CHANGED=NO
```
