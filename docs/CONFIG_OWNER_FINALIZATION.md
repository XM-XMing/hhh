# PY7-A configuration owner finalization

Date: 2026-08-29

The owner rule is:

```text
protocol constants -> protocol/constants.py
algorithm defaults -> domain dataclass/config
experiment values -> config/pre_bc.yaml
CLI -> explicit override only
shell -> no algorithm/runtime default ownership
resolved values -> persisted by the existing manifest/checkpoint/summary contracts
```

The requested duplicate owner count is zero after the targeted closure.  The
task-contract metadata blocker is now closed by an explicit version boundary:
V1 remains byte-for-byte frozen for historical P3/AWAC/SAC artifacts, while
new formal Pre-BC artifacts use V2 and include `max_primitive_steps` in their
canonical metadata hash.

## Owner matrix

| Parameter | Previous competing declarations | Canonical owner | YAML value | CLI default | Python constant/dataclass | Shell default | Resolved value |
|---|---|---|---:|---:|---|---|---|
| Managed ports | launcher/profile arithmetic and worker projections | `runtime/ports.py::MANAGED_TRAINING_PORT_DEFAULTS` + `WorkerRuntimeSpec` | — | profile/env projection | `11621/10553/12554`, stride `20` | none; env may explicitly override | unchanged managed profile |
| Legacy direct ports | direct runtime profile | `runtime/ports.py::LEGACY_COLLECTION_PORT_DEFAULTS` | — | explicit direct profile | `11321/10253/12254`, stride `20` | none | unchanged legacy diagnostic profile |
| `max_steps` | mission audit, collector, evaluation and RL consumers | `contracts/task.py::DEFAULT_MAX_PRIMITIVE_STEPS` and `task_contract_metadata` | `45` | `45` | `DEFAULT_MAX_PRIMITIVE_STEPS=45` | explicit `EVAL_REPRO_MAX_STEPS` only | `max_primitive_steps=45` in formal V2 |
| Primitive frame count | protocol/runtime launch literals and compatibility aliases | `protocol/constants.py::PRIMITIVE_FRAME_COUNT` | — | `25` through protocol aliases | `PRIMITIVE_FRAME_COUNT=25` | managed wrapper reads the Python protocol owner | `25` |
| Protocol/schema version | Python, Bridge and Unity protocol implementations | Python `protocol/constants.py`; C++/C# are wire implementations | — | `4` | `PROTOCOL_VERSION=4`, `SCHEMA_VERSION=4` | no Pre-BC numeric owner after PY7 | `4` |
| Teacher beam depth | CLI and Teacher construction sites | `teacher/policy.py::TeacherConfig` | — | `3` | `beam_depth=3` | none | `3` unless an explicit CLI override is supplied |
| Teacher beam width | CLI and Teacher construction sites | `TeacherConfig` | — | `8` | `beam_width=8` | none | `8` |
| Teacher beam branching | CLI and Teacher construction sites | `TeacherConfig` | — | `4` | `beam_branching=4` | none | `4` |
| Teacher beam discount | CLI and Teacher construction sites | `TeacherConfig` | — | `0.95` | `beam_discount=0.95` | none | `0.95` |
| Collision radius | depth-mask CLI and safety construction | `safety.depth_safety::DepthSafetyConfig` | — | `0.40` | `collision_radius_m=0.40` | none | `0.40 m` |
| Depth slack | depth-mask CLI and safety construction | `DepthSafetyConfig` | — | `0.08` | `depth_slack_m=0.08` | none | `0.08 m` |
| Path sample stride | depth-mask CLI and safety construction | `DepthSafetyConfig` | — | `4` | `path_sample_stride=4` | none | `4` |
| Patch radius | depth-mask CLI and safety construction | `DepthSafetyConfig` | — | `2` | `patch_radius_px=2` | none | `2 px` |
| Maximum patch radius | depth-mask CLI and safety construction | `DepthSafetyConfig` | — | `14` | `max_patch_radius_px=14` | none | `14 px` |
| Mission candidate count | mission CLI and launch examples | `config/pre_bc.yaml::mission.candidate_count` | `2,000,000` | `2,000,000` | no second numeric owner | none | `2,000,000` when not explicitly overridden |
| Mission seed | mission CLI and launch examples | `config/pre_bc.yaml::mission.seed` | `55` | `55` | no second numeric owner | none | `55` |
| Required passing missions | audit CLI and launch examples | `config/pre_bc.yaml::mission.required_passing` | `100,000` | `100,000` | no second numeric owner | none | `100,000` |
| Route candidate factor | mission CLI and launch examples | `config/pre_bc.yaml::mission.route_candidate_factor` | `1.15` | `1.15` | no second numeric owner | none | `1.15` |
| Route workers | mission CLI and launch examples | `config/pre_bc.yaml::mission.route_workers` | `20` | `20` | no second numeric owner | none | `20` |
| Teacher Label workers | `label_teacher_rollouts.py` and formal relabel command | `config/pre_bc.yaml::label.workers` | `20` | `20` | no second numeric owner | none | `20` |
| Depth-mask workers | `generate_depth_action_masks.py` and formal mask command | `config/pre_bc.yaml::depth_mask.workers` | `20` | `20` | no second numeric owner | none | `20` |
| Collection workers | parallel supervisor and shell examples | `config/pre_bc.yaml::collection.workers` | `20` | resolved default `20` | `pre_bc_value("collection", "workers")` | none | `20` |
| Target accepted | collection supervisor and shell examples | `config/pre_bc.yaml::collection.target_accepted` | `10,000` | environment/explicit override | `pre_bc_value("collection", "target_accepted")` | none | `10,000` formal default; smoke explicitly used `0` |
| BC epochs | BC CLI and training call sites | `bc.trainer::BCTrainingConfig` | — | `40` | `epochs=40` | none | `40` |
| BC batch size | BC CLI and training call sites | `BCTrainingConfig` | — | `128` | `batch_size=128` | none | `128` |
| BC learning rate | BC CLI and training call sites | `BCTrainingConfig` | — | `3e-4` | `lr=3e-4` | none | `3e-4` |
| BC weight decay | BC CLI and training call sites | `BCTrainingConfig` | — | `1e-4` | `weight_decay=1e-4` | none | `1e-4` |

The existing bounded smoke intentionally passed explicit values such as
`epochs=1`, `batch_size=16`, `MAX_STEPS=3`, and `TARGET_ACCEPTED=0`.  Those are
experiment overrides and do not create owners or change the formal defaults.

## Port, task, and frame status

```text
PORT_OWNER_UNIQUE=YES
TASK_MAX_STEPS_OWNER_UNIQUE=YES
PRIMITIVE_FRAME_COUNT_OWNER_UNIQUE=YES
TASK_CONTRACT_OWNER_UNIQUE=YES
FORMAL_PRE_BC_TASK_CONTRACT=V2
LEGACY_V1_FORMAL_CALLER_COUNT=0
```

The task-contract closure is:

```text
TASK_CONTRACT_V1_SHA256=5862af0f354c408d74cf4904dc7ecf609944553443ae1128097136be979fd30a
TASK_CONTRACT_V2_SCHEMA_VERSION=2
TASK_CONTRACT_V2_SHA256=2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df
MAX_PRIMITIVE_STEPS_IN_METADATA=YES
MAX_STEPS_CHANGE_CHANGES_SHA=PASS
```

V1 is exposed only through `legacy_task_contract_v1_metadata()` and
`legacy_task_contract_v1_sha256()` for explicit historical validation.  Formal
Pre-BC callers use `task_contract_fields()` from
`python/planning/contracts/task.py`; missing schema, unknown schema, missing
max, SHA mismatch, and V1 input fail closed.  The frozen AWAC/SAC path keeps
its explicit V1 identity and was not changed.
