# C4.1 — Depth Safety Production Cutover

Date: 2026-08-27

This is a P-only controlled cutover. The original tree remains the read-only
behavior baseline:

~~~
O = /home/xm/XM/xm_ws/src/planning
P = /home/xm/XM/src
~~~

No Unity, Bridge/protocol, Mission semantics, Teacher scoring algorithm,
BC/AWAC math, formal data collection, or depth-mask artifact provenance
producer was changed in C4.1. Python shared-VoxelMap binding remains a C5
item. No commit was created.

## Gate result

~~~
C4_1_DEPTH_SAFETY_PRODUCTION_CUTOVER=PASS
FORMAL_DEPTH_SAFETY_OWNER=planning.safety.depth_safety.local_depth_action_mask
FORMAL_NATIVE_IMPLEMENTATION=planning_depth_safety -> libplanning_depth_safety.so
PRODUCTION_IMPLEMENTATION_COUNT=1
PYTHON_PRODUCTION_FALLBACK=NO
NATIVE_DEPTH_SAFETY_FAIL_CLOSED=PASS
DEPTH_SAFETY_OWNER_UNIQUE=YES
DEPTH_CONFIG_OWNER_UNIQUE=YES
MASK_BIT_PARITY=PASS
DEPTH_MASK_GENERATOR_PARITY=PASS
POLICY_MASKED_ACTION_PARITY=PASS
TEACHER_DEPTH_SAFETY_PARITY=PASS
COLLISION_REGRESSION=PASS
GLOBAL_ROUTE_REGRESSION=PASS
P_DEPTH_SAFETY_TARGET_BUILD=PASS
P_FULL_BUILD=BLOCKED_BY_BRIDGE_PROTOCOL
DEPTH_MASK_ARTIFACT_PROVENANCE=NOT_HANDLED
ALGORITHM_CHANGED=NO
NEXT_PHASE=C5 PYTHON SHARED VOXELMAP AND NATIVE BINDING
~~~

## Owner and production call graph

The P public numeric seam is
planning.safety.depth_safety.local_depth_action_mask. It owns backend
resolution and is the only production module that knows the native library
path, ctypes handle, C ABI signature, and native buffer layout. The C++
implementation is the one formal algorithm implementation:

~~~
production caller
    -> planning.safety.depth_safety.local_depth_action_mask
    -> planning_depth_safety / planning_depth_safety_mask
~~~

The NumPy loop in the same module is retained only behind the explicit
PLANNING_DEPTH_SAFETY_BACKEND=python plus
PLANNING_DEPTH_SAFETY_REFERENCE=1 characterization seam. It is not an
automatic fallback and is not counted as a production implementation.

| CALLER | OLD DEPENDENCY | NEW CANONICAL OWNER | DUPLICATE LOGIC REMOVED |
| --- | --- | --- | --- |
| Depth Mask Generator (python/planning/safety/depth_action_masks.py and its script wrapper) | P moved generator/worker; O equivalent was planning.depth_action_masks | planning.safety.depth_safety.local_depth_action_mask through depth_action_mask_worker | None; generator remains orchestration and artifact serialization only |
| Unity runtime UnityForestEnv.get_action_mask | O planning.depth_safety; P moved local helper | planning.safety.depth_safety.local_depth_action_mask | None; runtime constructs a resolved DepthSafetyConfig and does not calculate mask geometry |
| Policy evaluator and counterfactual depth audit | O planning.cli.evaluate_policy_unity; P planning.evaluation.policy_evaluator | planning.safety.depth_safety.local_depth_action_mask | None; evaluator is a caller/audit consumer, not a native binding |
| Teacher collection | Environment action-mask path; no direct duplicate numeric helper in P Teacher code | UnityForestEnv.get_action_mask -> canonical owner when depth safety is enabled | None; Teacher scoring and selection were not changed |
| Teacher relabel / saved-mask consumption | DepthActionMaskStore/saved mask artifacts | DepthActionMaskStore for storage; generation path -> canonical owner | None; relabel does not recompute depth geometry |
| AWAC/RL training and evaluation consumers | full_depth_safety_mask and resolved run/checkpoint config | Environment or saved mask producer -> canonical owner | None; RL consumes masks and carries config, it does not implement depth geometry |
| DepthActionMaskStore and diagnostics | storage or diagnostic patch helpers | Not a numeric production owner | None; retained because these are separate storage/diagnostic contracts |

The static P audit found no production module other than
planning.safety.depth_safety containing planning_depth_safety_mask or
_NativeDepthSafety. Business callers import the public seam and do not
reference a native path or pointer layout.

## Fail-closed contract

| Condition | P behavior |
| --- | --- |
| Native library missing | DEPTH_SAFETY_NATIVE_UNAVAILABLE, explicit failure |
| Native symbol missing | DEPTH_SAFETY_NATIVE_SYMBOL_MISSING, explicit failure |
| Native call returns non-zero | DEPTH_SAFETY_NATIVE_EXECUTION_FAILED, explicit failure |
| Invalid depth shape | explicit ValueError at the public seam |
| auto, cpp, native, cpp_native | resolves to cpp_native; no fallback |
| python without reference opt-in | PYTHON_REFERENCE_DEPTH_SAFETY_REQUIRES_EXPLICIT_DEBUG |
| python with explicit reference opt-in | reference characterization only |
| Unknown backend | UNSUPPORTED_DEPTH_SAFETY_BACKEND |

There is no production --depth-safety-backend selector in the P generator or
policy-evaluation CLI. The compatibility environment aliases are resolved
inside the canonical owner. auto is native, and native failure remains fatal.

## Configuration ownership

The resolved numeric geometry is represented by the P
planning.safety.depth_safety.DepthSafetyConfig. The policy runtime contract
(planning.contracts.policy_runtime) validates and carries the four numeric
checkpoint values for compatibility; EnvConfig, evaluator arguments, and
training arguments are input/serialization carriers, not independent mask
implementations. The action width/order comes from the Motion Primitive
Library and the policy feature contract. Image dimensions and the pinhole
conversion convention are owned by the runtime depth observation contract and
camera_intrinsics_from_fov.

Current resolved values were preserved:

~~~
collision_radius_m = 0.40
depth_slack_m = 0.08
path_sample_stride = 4
max_patch_radius_px = 14
action_count = 105
camera FOV = 87.0 deg horizontal, 58.0 deg vertical
runtime valid range = 0.30 <= depth < 5.95 m
~~~

No new independent default or geometry helper was introduced in C4.1.
The four package export fields make the formal native identity explicit:
depth_safety_backend=cpp_native, depth_safety_contract_id=cpp_depth_patch,
depth_safety_library=libplanning_depth_safety.so, and
depth_safety_source_id=planning_depth_safety_cpp17.

## Duplicate scan and deletion ledger

No source file was deleted. The scan classified the apparent duplicates as
separate contracts:

| Candidate | Decision | Reason |
| --- | --- | --- |
| python/planning/safety/depth_safety.py NumPy branch | RETAIN | explicit O/P test/debug reference; no production fallback |
| python/planning/safety/depth_mask.py | RETAIN | saved-mask storage and normalized-depth reconstruction, not mask geometry |
| python/planning/diagnostics/depth_correctness.py | RETAIN | diagnostic patch inspection, not a production action-mask implementation |
| python/planning/contracts/policy_runtime.py | RETAIN | checkpoint numeric contract and compatibility resolution |
| python/planning/rl/train_sac.py depth fields | RETAIN | RL run/checkpoint configuration carrier; RL math and consumers were out of scope |
| O depth modules | RETAIN | read-only source baseline; no migration deletion is permitted |

~~~
DELETED_FILES=0
DELETION_LEDGER=NO_SAFE_PRODUCTION_DUPLICATES_DELETED
~~~

## Exact parity and regression evidence

The deterministic C4 comparator is
tests/compare_depth_safety_parity.py. It ran O and P in isolated processes
using the same temporary MPL and depth fixtures. It compared masks,
diagnostics, minimum clearances, projection tables, action order, generator
row mapping, policy masked logits/actions, and Teacher safety outputs.

~~~
synthetic edge frames = 18 x 105 actions
production-like frames = 1000 x 105 actions
production fixture SHA256 = 7d392527435b96310bd02b104bb5a581a9f0b6ccdb45dd9f0ce3107a8d6a080d
combined fixture SHA256 = 542d836acbc57e149cdc48e1bc599d884449489848450346975b0a7e0b67d9c1
MPL fixture SHA256 = f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d
MASK_BIT_PARITY=PASS
MIN_CLEARANCE_PARITY=PASS
VALID_COUNT_PARITY=PASS
EMPTY_MASK_PARITY=PASS
ACTION_ORDER_PARITY=PASS
SYNTHETIC_DEPTH_PARITY=PASS
PRODUCTION_LIKE_DEPTH_PARITY=PASS
DEPTH_MASK_GENERATOR_PARITY=PASS
POLICY_MASKED_ACTION_PARITY=PASS
TEACHER_DEPTH_SAFETY_PARITY=PASS
~~~

Frozen regressions were rerun after the cutover:

~~~
COLLISION_REGRESSION=PASS
  121 numeric cases exact; 100 candidate rows exact; 100 Teacher masks exact
GLOBAL_ROUTE_REGRESSION=PASS
  127 synthetic routes; 1000 production-like routes;
  1000 mission acceptance/order rows; 100 Teacher route/action/mask rows
~~~

The machine-readable depth result is
/tmp/xmflight_depth_safety_c4_cutover/depth_safety_parity.json. The collision result
is /tmp/xmflight_collision_c2/c4_collision_parity.json; the route summary is
/tmp/xmflight_global_route_c3_c4/c3_global_route_parity_summary.json.

## Performance

The same three-run, 1000-frame, 105-action native benchmark was rerun:

| Metric | O | P |
| --- | ---: | ---: |
| depth-safety throughput (actions/sec) | 259187.179987 | 270011.951375 |
| frames/sec | 2468.449333 | 2571.542394 |
| mean call (ms) | 0.404764 | 0.388547 |
| p50 call (ms) | 0.461788 | 0.462752 |
| p95 call (ms) | 0.473203 | 0.475622 |
| mean initialization (s) | 0.016592 | 0.016789 |
| maximum RSS (KB) | 150948 | 150948 |

~~~
P_DEPTH_SAFETY_THROUGHPUT=270011.951375 actions/sec
O_DEPTH_SAFETY_THROUGHPUT=259187.179987 actions/sec
DEPTH_SAFETY_DELTA_PERCENT=+4.176430%
P_MEETS_95_PERCENT_GATE=YES
~~~

Allocation counts were not instrumented. The benchmark reported 1000 native
crossings per run for both trees. No throughput claim is made for a CUDA
depth-safety backend.

## Build, package, and install audit

P's CMake declares one planning_depth_safety shared-library target from
src/geometry/depth_safety.cpp, compiles it under the project C++17 Release
configuration, and includes it in install(TARGETS ...). The public header is
include/planning/geometry/depth_safety.hpp; the runtime library is
libplanning_depth_safety.so. The package export records the native backend
and contract identity listed above. The source-only generated
data/motion_primitives directory remains conditionally skipped when absent;
no artifact was fabricated.

~~~
P_DEPTH_SAFETY_TARGET_BUILD=PASS
~~~

The generated install script contains the depth library install command, but a
full cmake --install invocation in conda activate xm stopped earlier in the
generated Python install at setup.py --install-layout=deb (the active
setuptools does not recognize that option). Therefore runtime package
installation is not claimed complete in this phase. The full P build still
stops only at the unchanged Bridge/protocol ROS_WARN,
ROS_WARN_THROTTLE, and ROS_ERROR_THROTTLE declarations:

~~~
P_FULL_BUILD=BLOCKED_BY_BRIDGE_PROTOCOL
~~~

Bridge repair and install-system repair are not C4.1 work.

## Artifact provenance boundary

The numerical owner cutover intentionally did not change depth-mask artifact
metadata. The following remain a later Python/data-pipeline concern:

~~~
DEPTH_MASK_ARTIFACT_PROVENANCE=NOT_HANDLED
observation_contract = NOT_HANDLED
observation_source = NOT_HANDLED
reliable_rows = NOT_HANDLED
legacy_rows = NOT_HANDLED
~~~

## Modified files in C4.1

~~~
package.xml
tests/test_depth_safety_production_cutover.py
docs/DEPTH_SAFETY_PRODUCTION_CUTOVER.md
docs/DEPTH_SAFETY_PARITY.md
docs/CXX_FOUNDATION_MIGRATION.md
docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md
~~~

No O file, Unity file, BC/AWAC implementation, Bridge/protocol source, or
formal data artifact was modified. C5 is the next phase; it was not started.
