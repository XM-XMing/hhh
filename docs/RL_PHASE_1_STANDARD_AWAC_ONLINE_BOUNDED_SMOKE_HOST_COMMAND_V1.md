# RL Phase 1 Standard AWAC Online Bounded Smoke Host Command V1

Date: 2026-09-05

This is a read-only certification of the Standard AWAC online handoff CLI and
the host-command boundary. No ROS master, Unity, Bridge, socket capability
probe, Standard AWAC runtime, Dev100 evaluation, Final300 evaluation, or
production Actor update was executed. The CPU-only learner contract tests did
exercise their synthetic Actor-update math path; that was not a runtime or
training update. The only source-tree changes for this task are this report
and its machine-readable companion.

## Result

```text
RL_PHASE_1_STANDARD_AWAC_ONLINE_BOUNDED_SMOKE_HOST_COMMAND_V1=FAIL
STANDARD_AWAC_ONLINE_CLI_READY=NO
STANDARD_AWAC_HANDOFF_ALLOWED=YES
RUNTIME_EXECUTED=NO
ACTOR_UPDATE_EXECUTED=NO
PRODUCTION_ACTOR_UPDATE_EXECUTED=NO
UNIT_LEARNER_ACTOR_UPDATE_PATH=EXERCISED
```

The Calibration PASS handoff itself remains valid. The blocker is the next
runtime stage: the current `awac_training` branch does not implement a
bounded online loop. It opens the existing replay, samples one batch, calls
one Actor-enabled learner update, saves `checkpoint_latest.pt`, and exits.
It does not start the managed runtime, collect an environment transition,
append an `AWAC_ONLINE` row, enforce the online cap, run Phase-1 milestones,
or write the requested smoke diagnostics.

This is a hard blocker for a Standard AWAC *online* smoke. Running the current
`awac_training` command directly would therefore be an invalid one-update
offline action, not the requested smoke; no such command was run or approved
as a smoke command.

## Frozen handoff inputs

```text
BC checkpoint:
data/teach/2026_6w/bc_training/checkpoint_best_soft.pt
BC SHA256:
ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2

Calibration PASS checkpoint:
data/awac/awac_bc60k_formal_critic_calibration_v6/checkpoint_calibration_pass.pt
Calibration PASS checkpoint SHA256:
bb9433d8d9fb2fde2e2148e84a4058c2b328b9d676608d38c00962f821c3d8de

Calibration replay:
data/awac/awac_bc60k_formal_critic_calibration_v6/replay
Calibration replay size:
5142
Calibration behavior source:
BC_CALIBRATION=5142, AWAC_ONLINE=0
Calibration replay metadata phase:
critic_calibration
```

The current replay metadata is a valid Calibration artifact with the exact
endpoint observation contract, zero legacy rows, task contract SHA
`2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df`, and MPL
contract SHA
`f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d`. It is
not modified by this task.

## CLI audit

The real CLI choices are:

```text
--phase {critic_calibration,awac_training}
--bc-checkpoint PATH
--train-index PATH
--out-dir PATH
--resume-checkpoint PATH
--replay-dir PATH
--env-workers N
--worker-spec-file PATH
--reliable-v4
--total-env-steps N
--phase1-online-transition-cap N
```

The parser also exposes the frozen optimizer, replay, runtime timeout, and
depth-mask flags. The exact help output was read under `conda activate xm`.
`audit_awac_replay.py` has only the real flags `--replay-dir` and optional
`--output`.

The CLI has no separate `--online-transition-budget` flag. Although
`--total-env-steps` exists, the current Standard branch does not use it to
drive an environment loop. `--phase1-online-transition-cap` is persisted in
the Phase-1 contract, whose current milestone builder requires the cap to
reach the first 10,000-transition milestone; setting it to the required 1,000
smoke transitions is therefore not a valid current contract construction.

## Source-path evidence

| Surface | Current evidence | Assessment |
| --- | --- | --- |
| Standard phase name | `trainer.py:119-121` accepts CLI value `awac_training`; the handoff state is `ACTOR_ENABLED_STANDARD_AWAC` | Present |
| Handoff gate | `trainer.py:248-260` requires a resume checkpoint and replay; `trainer.py:2288-2300` validates the handoff and enables the Actor | PASS at construction boundary |
| Managed runtime | `trainer.py:1304-1316` starts it only from `_run_online_calibration`; that function is selected at `trainer.py:2215-2259` only for `critic_calibration` with `train_index` | Missing for Standard |
| Online producer | `calibration_runtime.py:1191-1202` writes `BehaviorSource.BC_CALIBRATION`; no Standard producer writes `BehaviorSource.AWAC_ONLINE` | Missing |
| Online loop/budget | `trainer.py:2313-2315` samples one batch; `trainer.py:2408-2420` calls one `learner.update` and saves one checkpoint | Missing |
| Phase-1 milestones | `phase1.py:502-564` defines the hook, but current `trainer.py` has no call from the Standard branch | Not integrated |
| Standard summary | Current Standard branch prints only `AWAC_UPDATE=PASS`; it has no online counters, behavior-source counts, Actor before/after SHA, or objective diagnostic artifact | Missing |
| Independent replay handoff | `--replay-dir` is an existing replay path used for handoff validation; no current Standard owner clones the V6 replay and changes its behavior-source phase before appending online rows | Missing |

The existence of the `AWAC_ONLINE` enum value is not evidence of a producer:
`planning.awac.interaction` defines the value, while the only current runtime
transition construction uses `BC_CALIBRATION`.

## Frozen Standard AWAC configuration

The current parser defaults and the requested frozen values agree:

```text
actor_depth_lr=1e-6
actor_head_lr=1e-5
actor_vector_lr=3e-6
critic_depth_lr=1e-5
critic_head_lr=1e-4
critic_vector_lr=1e-5
gamma=0.99
tau=0.005
awac_temperature=2.0
awac_weight_max=20.0
awac_bc_kl_weight=0.05
bc_kl_hard_budget=0.10
bc_kl_recovery_weight=1.0
critic_cql_weight=0.05
gradient_clip_norm=5.0
reward_scale=0.10
```

`BC_KL_MODE=FIXED`. Confidence, adaptive BC KL, and primitive-neighbor
algorithms are not implemented and were not added.

## Smoke boundary

The requested smoke parameters remain recorded, but no smoke artifact was
created:

```text
workers=2
online transition budget=1000
output=data/awac/smoke/standard_awac_online_w2_v1
Dev100=not executed
Final300=not executed
```

The current filesystem has 138 GiB free on the Planning volume. The prior
measured 50K replay apparent size is about 1.399 GiB, so the conservative
static smoke disk preflight is PASS with a 5 GiB ceiling. This is only a
capacity check; it is not evidence that the blocked runtime can complete.

## Verification performed

All commands that executed a Python script were preceded by
`conda activate xm`.

```text
python scripts/train_awac.py --help                         PASS
python scripts/audit_awac_replay.py --help                 PASS
python -m compileall -q python scripts                      PASS
python tests/run_pytest_unit.py /tmp/rl-phase1-standard-awac-online-unit.xml
                                                               561 passed, 2 skipped, 0 failed
```

The unit run selected 561 CPU-only unit tests and deselected 274 non-unit
tests. A small subset of those tests exercises the learner's synthetic Actor
update path; it does not create a runtime, environment transition, replay
append, or training run. It did not start ROS, Unity, Bridge, or a socket
runtime. No full
pytest run was attempted because this task explicitly forbids runtime
execution and socket capability probing.

## Host commands

Because the online owner is absent, the first command is deliberately
fail-closed. It verifies the immutable inputs and exits before copying the
replay or invoking `train_awac.py` on the current tree. The train command is
shown only behind that source-level guard so a host operator cannot
accidentally run the current one-update Standard branch as an online smoke.

### HOST_STANDARD_AWAC_W2_SMOKE

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
source /opt/ros/noetic/setup.bash
source /home/xm/XM/xm_ws/devel/setup.bash
cd /home/xm/XM/xm_ws/src/planning
set -euo pipefail
ROOT=/home/xm/XM/xm_ws/src/planning
V6="$ROOT/data/awac/awac_bc60k_formal_critic_calibration_v6"
BC="$ROOT/data/teach/2026_6w/bc_training/checkpoint_best_soft.pt"
MISSIONS="$ROOT/data/teach/2026_6w/missions.csv"
WORKER_SPEC="$ROOT/data/teach/2026_6w/rollouts/worker_runtime_specs.json"
OUT="$ROOT/data/awac/smoke/standard_awac_online_w2_v1"
REPLAY_SRC="$V6/replay"
REPLAY="$OUT/replay"
PASS_CHECKPOINT="$V6/checkpoint_calibration_pass.pt"
BC_SHA=ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2
PASS_SHA=bb9433d8d9fb2fde2e2148e84a4058c2b328b9d676608d38c00962f821c3d8de
MPL_SHA=f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d
TASK_SHA=2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df
test -f "$BC"
test -f "$MISSIONS"
test -f "$WORKER_SPEC"
test -f "$PASS_CHECKPOINT"
test "$(sha256sum "$BC" | awk '{print $1}')" = "$BC_SHA"
test "$(sha256sum "$PASS_CHECKPOINT" | awk '{print $1}')" = "$PASS_SHA"
if ! rg -q 'def _run_standard_awac_online|class StandardAWACOnline|behavior_source.*BehaviorSource\.AWAC_ONLINE' python/planning/awac/trainer.py python/planning/awac/calibration_runtime.py; then
  echo 'STANDARD_AWAC_ONLINE_CLI_READY=NO' >&2
  echo 'BLOCKER=the current awac_training branch has no online transition producer or bounded online loop' >&2
  exit 2
fi
test ! -e "$OUT"
mkdir -p "$REPLAY"
cp --archive --reflink=auto "$REPLAY_SRC/." "$REPLAY/"
python scripts/train_awac.py \
  --phase awac_training \
  --bc-checkpoint "$BC" \
  --train-index "$MISSIONS" \
  --out-dir "$OUT" \
  --resume-checkpoint "$PASS_CHECKPOINT" \
  --replay-dir "$REPLAY" \
  --mpl-contract-sha256 "$MPL_SHA" \
  --total-env-steps 1000 \
  --max-steps 45 \
  --env-workers 2 \
  --worker-spec-file "$WORKER_SPEC" \
  --reliable-v4 \
  --device auto \
  --seed 55 \
  --cpu-threads 1 \
  --replay-capacity 50000 \
  --batch-size 128 \
  --learning-starts 5000 \
  --actor-learning-starts 8000 \
  --critic-burnin-updates 2000 \
  --actor-update-interval 4 \
  --updates-per-step 0.50 \
  --warmup-temperature 1.0 \
  --reward-scale 0.10 \
  --gamma 0.99 \
  --tau 0.005 \
  --actor-head-lr 1e-5 \
  --actor-vector-lr 3e-6 \
  --actor-depth-lr 1e-6 \
  --critic-head-lr 1e-4 \
  --critic-vector-lr 1e-5 \
  --critic-depth-lr 1e-5 \
  --critic-cql-weight 0.05 \
  --awac-temperature 2.0 \
  --awac-weight-max 20.0 \
  --awac-bc-kl-weight 0.05 \
  --awac-trust-tail-top-k 16 \
  --bc-kl-hard-budget 0.10 \
  --bc-kl-recovery-weight 1.0 \
  --gradient-clip-norm 5.0 \
  --checkpoint-interval-steps 5000 \
  --log-interval-steps 100 \
  --reset-settle 0.30 \
  --reset-timeout 10.0 \
  --worker-ready-timeout 30.0 \
  --post-wait 0.0 \
  --max-sensor-skew-ms 80.0 \
  --depth-mask-collision-radius 0.40 \
  --depth-mask-slack 0.08 \
  --depth-mask-sample-stride 4 \
  --depth-mask-max-patch-radius-px 14
```

The command is not a current smoke-ready command: its guard exits on the
current source. In particular, `--total-env-steps 1000` cannot make the
current Standard branch online; the branch does not consume that value as a
loop bound.

### HOST_STANDARD_AWAC_W2_REPLAY_AUDIT

This command is a post-smoke audit only. It is safe for the new smoke
directory and never points the audit at V6.

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
source /opt/ros/noetic/setup.bash
source /home/xm/XM/xm_ws/devel/setup.bash
cd /home/xm/XM/xm_ws/src/planning
set -euo pipefail
ROOT=/home/xm/XM/xm_ws/src/planning
OUT="$ROOT/data/awac/smoke/standard_awac_online_w2_v1"
REPLAY="$OUT/replay"
AUDIT="$OUT/replay_audit.json"
test -d "$REPLAY"
python scripts/audit_awac_replay.py --replay-dir "$REPLAY" --output "$AUDIT"
python - "$AUDIT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
counts = {str(key): int(value) for key, value in report.get("behavior_source_counts", {}).items()}
checks = {
    "status": report.get("status") == "PASS",
    "legacy_replay_transition_count": int(report.get("legacy_replay_transition_count", -1)) == 0,
    "privileged_field_count": int(report.get("privileged_field_count", -1)) == 0,
    "bc_calibration_rows": counts.get("0", 0) > 0,
    "awac_online_rows": counts.get("1", 0) > 0,
}
for name, value in checks.items():
    print("{}={}".format(name.upper(), "PASS" if value else "FAIL"))
if not all(checks.values()):
    raise SystemExit(2)
PY
```

### HOST_STANDARD_AWAC_W2_SMOKE_SUMMARY

This is a read-only, fail-closed summary reader for a completed smoke. It
accepts the current learner metric names and does not write any artifact. It
exits non-zero when the required online summary, checkpoint counters, or
replay audit evidence is absent.

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
source /opt/ros/noetic/setup.bash
source /home/xm/XM/xm_ws/devel/setup.bash
cd /home/xm/XM/xm_ws/src/planning
set -euo pipefail
ROOT=/home/xm/XM/xm_ws/src/planning
OUT="$ROOT/data/awac/smoke/standard_awac_online_w2_v1"
python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()

def load_json(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def find_value(value, names):
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        for child in value.values():
            found = find_value(child, names)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_value(child, names)
            if found is not None:
                return found
    return None

summary = None
for candidate in (root / "summary.json", root / "smoke_summary.json", root / "training_summary.json"):
    summary = load_json(candidate)
    if summary is not None:
        break
audit = load_json(root / "replay_audit.json")
metadata = load_json(root / "replay" / "metadata.json")
checkpoint_path = next(
    (
        candidate
        for candidate in (
            root / "checkpoint_latest.pt",
            root / "checkpoint_last.pt",
            root / "checkpoint_awac_50k.pt",
            root / "checkpoint_awac_25k.pt",
            root / "checkpoint_awac_10k.pt",
        )
        if candidate.is_file()
    ),
    None,
)
checkpoint = None
if checkpoint_path is not None:
    import torch
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
if summary is None or audit is None or metadata is None or not isinstance(checkpoint, dict):
    print("SMOKE_SUMMARY_CERTIFICATION=FAIL")
    print("REQUIRED_SUMMARY_AUDIT_METADATA_CHECKPOINT=FAIL")
    raise SystemExit(2)

def find_sources(names):
    value = find_value(summary, names)
    if value is None:
        value = find_value(checkpoint, names)
    return value

for label, names in {
    "ONLINE_TRANSITIONS": ("online_transitions", "online_transition_count"),
    "ACTOR_UPDATE_COUNT": ("actor_update_count",),
    "CRITIC_UPDATE_COUNT": ("critic_update_count",),
    "ACTOR_SHA_BEFORE": ("actor_sha_before", "actor_state_sha_before", "bc_reference_fingerprint"),
    "ACTOR_SHA_AFTER": ("actor_sha_after", "actor_state_sha_after"),
    "TD_LOSS": ("critic_td_loss", "td_loss"),
    "Q_MEAN": ("q_mean",),
    "Q_STD": ("q_std",),
    "AWAC_WEIGHT_MEAN": ("awac_weight_mean",),
    "AWAC_WEIGHT_MAX": ("awac_weight_max",),
    "BC_KL": ("bc_kl",),
    "NAN_COUNT": ("nan_count",),
    "INF_COUNT": ("inf_count",),
}.items():
    print("{}={}".format(label, find_sources(names)))

source_counts = audit.get("behavior_source_counts", {})
print("CHECKPOINT_PATH={}".format(checkpoint_path))
print("BC_CALIBRATION_COUNT={}".format(source_counts.get("0", 0)))
print("AWAC_ONLINE_COUNT={}".format(source_counts.get("1", 0)))
print("OBSERVATION_CONTRACT={}".format(metadata.get("observation_contract")))
print("REPLAY_BEHAVIOR_SOURCE_PHASE={}".format(metadata.get("behavior_source_phase")))
print("CHECKPOINT_LAST_GENERATION_COUNT={}".format(len(tuple(root.glob("checkpoint_last.generation-*.pt")))))
cleanup = find_value(summary, ("cleanup", "process_cleanup", "runtime_cleanup", "runtime_close"))
print("RUNTIME_CLEANUP={}".format(cleanup))

required = {
    "online transitions": find_value(summary, ("online_transitions", "online_transition_count")),
    "actor updates": find_sources(("actor_update_count",)),
    "critic updates": find_sources(("critic_update_count",)),
    "actor before SHA": find_sources(("actor_sha_before", "actor_state_sha_before", "bc_reference_fingerprint")),
    "actor after SHA": find_value(summary, ("actor_sha_after", "actor_state_sha_after")),
    "TD loss": find_value(summary, ("critic_td_loss", "td_loss")),
    "Q mean": find_value(summary, ("q_mean",)),
    "AWAC weight mean": find_value(summary, ("awac_weight_mean",)),
    "BC KL": find_value(summary, ("bc_kl",)),
}
checks = {
    "summary_fields": all(value is not None for value in required.values()),
    "online_budget": int(required["online transitions"]) <= 1000 if required["online transitions"] is not None else False,
    "actor_updates_positive": int(required["actor updates"]) > 0 if required["actor updates"] is not None else False,
    "critic_updates_positive": int(required["critic updates"]) > 0 if required["critic updates"] is not None else False,
    "actor_changed": required["actor before SHA"] != required["actor after SHA"] if required["actor before SHA"] is not None and required["actor after SHA"] is not None else False,
    "source_mix": int(source_counts.get("0", 0)) > 0 and int(source_counts.get("1", 0)) > 0,
    "replay_pass": audit.get("status") == "PASS",
    "legacy_zero": int(audit.get("legacy_replay_transition_count", -1)) == 0,
    "nan_zero": find_sources(("nan_count",)) == 0,
    "inf_zero": find_sources(("inf_count",)) == 0,
}
print("SMOKE_SUMMARY_CERTIFICATION={}".format("PASS" if all(checks.values()) else "FAIL"))
for name, value in checks.items():
    print("CHECK_{}={}".format(name.upper(), "PASS" if value else "FAIL"))
if not all(checks.values()):
    raise SystemExit(2)
PY
```

These three commands are not executed by Codex. The first one is intentionally
blocked on the current source; the latter two are post-artifact commands and
cannot pass until a real Standard online producer writes the required smoke
artifacts.

## Boundary and next action

```text
BC_CHANGED=NO
CRITIC_CHANGED=NO
REWARD_CONTRACT_CHANGED=NO
REPLAY_CONTRACT_CHANGED=NO
TASK_CONTRACT_CHANGED=NO
OBSERVATION_CONTRACT_CHANGED=NO
MPL_CHANGED=NO
UNITY_CHANGED=NO
BRIDGE_CHANGED=NO
CONFIDENCE_ALGORITHM_IMPLEMENTED=NO
ADAPTIVE_BC_KL_IMPLEMENTED=NO
PRIMITIVE_NEIGHBOR_ALGORITHM_IMPLEMENTED=NO
COMMIT=NO
```

The next action is:

```text
FIX_STANDARD_AWAC_ONLINE_RUNTIME_BLOCKER
```

The fix must add a formal Standard online runtime owner that consumes the
validated Calibration PASS handoff, operates on an isolated replay snapshot,
records `AWAC_ONLINE` transitions, applies the 1,000-transition bound, calls
the existing Actor/Critic learner without changing its math, persists the
required diagnostics, and integrates the existing Phase-1 state-machine
milestone hooks. It is outside this command-generation task.
