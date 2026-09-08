# PY7-A common capability finalization

Date: 2026-08-28

This record covers only the canonical O tree,
`/home/xm/XM/xm_ws/src/planning`.  P and Unity were read-only references.  The
change closes the generic common seams needed by the Pre-BC pipeline while
retaining separate domain contracts where their canonical bytes or lifecycle
semantics differ.

## Counting rule

`DUPLICATE_COMMON_CAPABILITY_COUNT` counts the three generic implementation
families selected for PY7-A:

1. file SHA-256 wrappers;
2. canonical JSON serialization/hash wrappers;
3. atomic JSON replacement wrappers.

It does not count distinct semantic identities as duplicates.  Protocol
MessagePack bytes, source-tree semantic hashes, MPL contract hashes,
checkpoint/tensor fingerprints, normalizer fingerprints, CSV/NPZ writers, and
dataset-directory transactions therefore remain separate owners.

```text
DUPLICATE_COMMON_CAPABILITY_COUNT_BEFORE=3
DUPLICATE_COMMON_CAPABILITY_COUNT_AFTER=0
```

## Capability inventory

| Capability | Current implementations/callers | Canonical owner | Migration | Deletion candidates |
|---|---|---|---|---|
| File SHA-256 | BC trainer, BC mmap, runtime identity, parallel collection, reliable single-worker diagnostics, provenance, feature and RL readers | `python/planning/common/hashing.py::file_sha256` | Generic file reads now use the common helper | None; semantic source/checkpoint owners remain |
| Canonical JSON | Collection UTF-8 contract, provenance/task/feature identities, BC training config and lifecycle topology | `canonical_json_bytes` / `canonical_json_sha256`, with explicit `ensure_ascii` mode | Existing byte modes are preserved rather than silently conflated | None |
| Atomic JSON | Runtime identity/lifecycle/worker and existing mission/collection persistence paths | `python/planning/common/io.py::write_json_atomic` | `trailing_newline` makes the two existing output contracts explicit | None |
| Atomic CSV | Rollout and label artifacts | `planning.data.rollout` and domain label writers | Retained because CSV schema and row ordering are domain contracts | None |
| Atomic NPZ | Label and depth-mask artifacts | `planning.teacher.labeling` / `planning.safety.depth_action_masks` | Retained because NPZ payload and compression are artifact contracts | None |
| Temporary-file replace | Generic JSON/text/bytes writes | `common.io::_replace_bytes` | One generic replacement seam; output-local dataset/checkpoint replacement stays separate | None |
| Path resolution | Package, workspace and native-library lookup | `python/planning/common/paths.py` | Source/devel/install lookup remains centralized; explicit package-root arguments remain supported | None |
| Artifact identity | Rollout/provenance validation and frozen runtime identity | `contracts/pipeline_provenance.py`, `runtime/identity.py`, `runtime/bridge_identity.py` | Kept separate from generic hashing because required fields and failure policy differ | None |
| Config SHA | Collection resolved-config and merge-compatibility projections | `contracts/collection.py` plus common canonical bytes | Semantic projection remains owned by collection contracts | None |
| Checkpoint SHA | BC checkpoint file identity and RL checkpoint evidence | BC trainer file identity and protected RL fingerprint modules | Not merged with generic artifact hashing | None; RL is deferred |
| Source manifest SHA | Planning source tree and Bridge source manifest | `contracts.collection::source_tree_sha256` and `runtime.bridge_identity::bridge_source_manifest_sha256` | Kept as two explicit manifests because their file sets and identity contracts differ | None |

The retained private functions are semantic implementations, not generic
duplicates: tensor/normalizer hashing, MessagePack canonicalization, source
tree traversal, and artifact-specific atomic payload writers are intentionally
not deleted.

## Files changed by PY7-A

The common seam itself is in:

```text
python/planning/common/config.py
python/planning/common/hashing.py
python/planning/common/io.py
python/planning/common/__init__.py
python/planning/protocol/constants.py
```

The affected callers project existing behavior onto those seams in the
mission, Teacher, safety, BC, runtime, provenance, and diagnostic modules.
The protocol aliases `SCHEMA_VERSION`, `FRAME_COUNT`, and
`DEFAULT_REQUESTED_FRAME_COUNT` remain public compatibility names; their
numeric owner is `protocol/constants.py`.

No data array, row order, artifact schema, wire field, canonical byte stream,
or algorithm was changed.  No helper or source file met the deletion rule
(`caller=0`, formal entry=0, dynamic/public dependency=0, checkpoint/source
manifest dependency=0, and migrated tests), so no deletion was performed.

