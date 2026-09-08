# Accepted-only collection finalization

The formal collection report contains every committed attempt. Downstream
Teacher/BC stages consume the published `rollout_index.csv`, which is a clean
accepted-only selection and is not a claim that the complete runtime history
was error-free.

## Publication rules

For a formal reliable-exact shard, a row is eligible only when it is accepted
and has no collision, dead-end, hard-altitude, path-limit, execution, identity,
or observation-contract failure. Its NPZ is then loaded with
`planning.data.rollout.load_rollout_episode(path, validate=True)` and its
episode, transition, task, and observation provenance must agree with the
collection row. Rows are sorted by the canonical numeric `episode_id`; the
first `target_accepted` eligible rows are published.

Missing or invalid accepted NPZs fail closed. Collector errors attached only to
non-accepted attempts remain in the raw report and set
`runtime_quality_pass=false`, but do not block publication when the clean
accepted-only gate passes.

## Artifact cleanup

After successful accepted-only validation, NPZs belonging to non-accepted
attempts and unreferenced episode NPZ/temp/partial artifacts under worker
directories are removed. Worker journals, raw collection reports, structured
logs, and diagnostics are retained. Cleanup never edits accepted NPZ contents.

## Validation modes

`scripts/validate_teacher_collection.py` validates the accepted dataset by
default. Add `--require-runtime-quality` when the caller also requires the
complete collection history to have zero fatal collector errors and all legacy
runtime quality gates to pass.
