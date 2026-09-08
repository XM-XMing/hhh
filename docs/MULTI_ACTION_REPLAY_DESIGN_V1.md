# MULTI_ACTION_REPLAY_DESIGN_V1

## Scope

This is a diagnostic-only artifact for measuring state-internal action
ranking. It is not the formal AWAC replay and is not consumed by
`planning.awac.replay.AWACReplayBuffer`, the learner, BC, or any production
training command. The collector never calls an optimizer and never writes the
production replay.

The artifact is produced only from the reliable observation contract:

```text
reliable_exact_endpoint_snapshot
```

## Storage contract

Each directory contains `metadata.json` and `transitions.npz` with schema ID
`multi_action_replay_v1`. `transitions.npz` is loaded with
`allow_pickle=False` and has one row per state/action branch:

| field | required shape/meaning |
| --- | --- |
| `state_id` | non-empty content identity for the frozen source state |
| `mission_id` | canonical mission identity |
| `episode_id` | unique diagnostic candidate episode identity |
| `step_id` | source state step, non-negative |
| `route_id` | route identity or explicit `mission:<mission_id>` route fallback |
| `action_source` | one of `BC`, `TEACHER`, `NEIGHBOR`, `RANDOM` |
| `action` | legal action ID under `mask` |
| `mask` | `[action_count]` legal source-state action mask |
| `reward` | immediate environment reward for this branch |
| `state_vector` | source continuous observation vector |
| `state_depth` | source depth observation |
| `next_state_id` | content identity after the candidate action |
| `next_state_vector` | next continuous observation vector |
| `next_state_depth` | next depth observation |
| `next_mask` | `[action_count]` next legal-action mask |
| `done` | terminal flag for the candidate step |
| `transition_id` | unique transition identity |

The writer rejects unknown sources, non-finite arrays, inconsistent shapes,
actions outside the source mask, duplicate transition IDs, and duplicate
`(state_id, action)` pairs. The latter makes source ratios describe distinct
action branches; if Teacher and BC choose the same action, it is retained once
and the planner does not manufacture a duplicate row.

Metadata retains source checkpoint/index hashes, task/MPL identities when
provided, runtime identity, seed, and explicit isolation flags:

```json
{
  "production_replay": false,
  "training_consumed": false,
  "diagnostic_only": true
}
```

## Candidate planner and collector

For each of 500 failed-BC states, the collector deterministically plans:

1. frozen deterministic BC action;
2. exact same-state Teacher action, when a reliable recovery row matches;
3. up to two geometry-only MPL neighbors, filtered by the current mask;
4. up to three masked random actions, seeded per state.

Candidates are deduplicated by action ID. Every candidate is executed from a
fresh reset and replayed prefix, and is admitted only when the source state
content identity matches. A Q value is never used to choose an action. The
candidate transition stores the immediate reward; the same branch then follows
the frozen deterministic BC policy only to record a comparable full-episode
return and terminal outcome in `candidate_branches.jsonl`. Those continuation
actions are not additional multi-action candidate rows. The bounded default is
at most 3500 candidate episodes, 150,000 environment steps, and 4000 rows; the
publish gate is at least 500 distinct states and 2500 valid transitions.

The collector expects an already managed reliable-v4 worker spec. It does not
start Unity, Bridge, ROS, or a managed runtime itself. A non-empty output
directory is rejected, and failure to reach the scale gate does not publish a
`replay/` artifact.

Example command (execution is intentionally separate from implementation and
tests):

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
cd /home/xm/XM/xm_ws/src/planning
python scripts/collect_multi_action_replay.py \
  --worker-spec-file /path/to/one-worker-spec.json \
  --mission-index /path/to/missions.csv \
  --bc-index /path/to/bc/rollout_index.csv \
  --bc-dir /path/to/bc \
  --recovery-run /path/to/teacher/recovery/run \
  --bc-checkpoint /path/to/checkpoint_best_soft.pt \
  --out-dir /path/to/diagnostics/multi_action_replay_v1
```

Teacher recovery input is optional. Omitting it records no `TEACHER` rows;
it does not replace a missing match with a guessed action.

## Audit gate

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
cd /home/xm/XM/xm_ws/src/planning
python scripts/audit_multi_action_replay.py \
  --replay-dir /path/to/diagnostics/multi_action_replay_v1/replay \
  --min-states 500 \
  --min-transitions 2500 \
  --output /path/to/diagnostics/multi_action_replay_v1/audit.json
```

The audit reports state count, transition count, mean actions/state, unique
state/action ratio, source counts/ratios, terminal count, and structural or
scale failures. It is an evidence report only; `PASS` does not authorize
Critic or Actor training.

After collection, build same-state return pairs without a Critic:

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
cd /home/xm/XM/xm_ws/src/planning
python scripts/build_multi_action_pairwise_dataset.py \
  --branches /path/to/diagnostics/multi_action_replay_v1/candidate_branches.jsonl \
  --output /path/to/diagnostics/multi_action_replay_v1/multi_action_pairwise_dataset.csv \
  --audit-output /path/to/diagnostics/multi_action_replay_v1/pairwise_audit.json
```

Pairs use observed full-branch returns only; no Q value selects a good action.
Equal-return pairs are retained and excluded from effective random-baseline
accuracy. Runtime/invalid branches remain in `candidate_branches.jsonl` with a
failure reason and are never silently converted to valid transitions.

## Production boundary

No production defaults, model dimensions, reward, observation contract,
`AWACReplayBuffer`, checkpoint, or training gate are changed by this design.
The next experiment must explicitly convert the diagnostic artifact through a
separate reviewed producer if it is ever to become training data.
