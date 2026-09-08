# C4 — Depth safety parity and ownership

Date: 2026-08-27

The original tree is the read-only behavior baseline:

```text
O = /home/xm/XM/xm_ws/src/planning
P = /home/xm/XM/src
```

This phase covers only `depth observation -> primitive/action safety result`.
No Unity, Bridge, Teacher scoring, Mission semantics, BC/AWAC, observation
provenance producer, formal data collection, or Python shared-VoxelMap binding
was changed.

## Result

```text
C4_DEPTH_SAFETY_PARITY=PASS
STRUCTURE_CHANGED=YES
DEPTH_SAFETY_BEHAVIOR_CHANGED=NO
ALGORITHM_CHANGED=NO
P_DEPTH_SAFETY_PRODUCTION_CANDIDATE=YES
```

The only intentional runtime-policy change is failure handling in P: native
depth-safety initialization, symbol lookup, and execution are fail-closed.
Python reference execution requires an explicit test/debug opt-in.  The
numeric geometry, thresholds, loop order, outputs, and action order were not
changed.

## Owner and call graph map

| Concern | O owner | P owner | Classification |
| --- | --- | --- | --- |
| Public numeric entry | `planning.depth_safety.local_depth_action_mask` | `planning.safety.depth_safety.local_depth_action_mask` | `STRUCTURAL_REFACTOR` |
| Native implementation | `src/depth_safety.cpp` | `src/geometry/depth_safety.cpp` + `include/planning/geometry/depth_safety.hpp` | `EXACT_MOVE` plus header/namespace move |
| Projection table | `planning.depth_safety.primitive_depth_projection_table` | `planning.safety.depth_safety.primitive_depth_projection_table` | `EXACT_MOVE` |
| Saved-mask storage | `planning.depth_action_masks.DepthActionMaskStore` | `planning.safety.depth_mask.DepthActionMaskStore` | `STRUCTURAL_REFACTOR` |
| Mask generator | `planning.cli.generate_depth_action_masks` | `planning.safety.depth_action_masks` and `scripts/generate_depth_action_masks.py` | `STRUCTURAL_REFACTOR` |
| Worker | `planning.depth_action_mask_worker` | `planning.safety.depth_action_mask_worker` | `EXACT_MOVE` |
| Runtime caller | `planning.unity_env.UnityForestEnv.get_action_mask` | `planning.runtime.unity_env.UnityForestEnv.get_action_mask` | `STRUCTURAL_REFACTOR` |
| Evaluator caller | `planning.cli.evaluate_policy_unity` | `planning.evaluation.policy_evaluator` | `STRUCTURAL_REFACTOR` |
| Policy mask caller | `planning.bc_model.mask_logits` | `planning.bc.model.mask_logits` | `EXACT_MOVE` at public seam |

The formal P algorithm owner is the native `planning_depth_safety` C ABI,
called through the single `local_depth_action_mask` seam.  The NumPy loop is
retained only as an explicit reference implementation.  There is no formal
second algorithm owner in P.

Owner-count evidence:

```text
MPL_LOAD_COUNT_PER_PROCESS=1
DEPTH_CONFIG_OWNER_COUNT=1
ACTION_ORDER_OWNER_COUNT=1
PYTHON_REFERENCE_FORMAL_OWNER=NO
```

The callers do not own the C ABI pointer layout or shared-library path.

## Canonical semantic matrix

The values below were read from both current trees and exercised by the
parity runner; they are not copied from historical prompts.

| Contract | O/P result |
| --- | --- |
| Depth dtype/unit | `float32`, metric metres |
| Depth shape | exactly two-dimensional `[H,W]`; invalid shape raises |
| Input orientation | image row/column coordinates; no extra vertical flip |
| Body to optical | `(forward,left,up) -> (right,down,forward) = (-y,-z,x)` |
| Camera FOV | horizontal `87.0 deg`, vertical `58.0 deg` |
| Principal point | `cx=(W-1)/2`, `cy=(H-1)/2` |
| Projection rounding | `np.rint`/native table values from the same Python projection owner |
| Minimum forward | `0.45 m` |
| Path sample stride | `4`, with the final reference frame appended |
| Collision radius | `0.40 m` |
| Depth slack | `0.08 m` |
| Patch radius | minimum `2 px`, raw radius capped at `14 px` |
| Runtime valid range | finite and `0.30 <= depth < 5.95 m` |
| Invalid pixels | zero, negative, NaN, Inf, and out-of-range values are unknown/non-blocking |
| Patch boundary | clipped to image bounds; scan order is y then x |
| Blocking rule | minimum valid depth minus point depth `<= 0.40+0.08` |
| Action width/order | formal MPL owner, `105` actions, ascending action IDs |
| Empty mask | all-invalid visible patches remain valid; only observed blocking can reject |

The mask-generator CLI has a separate saved-depth threshold of `2.95 m`, as
specified by its current arguments.  It was not conflated with the runtime
sensor maximum or changed in C4.

## Failure policy

| Case | P formal behavior |
| --- | --- |
| native library missing | `DEPTH_SAFETY_NATIVE_UNAVAILABLE`, fail closed |
| native symbol missing | `DEPTH_SAFETY_NATIVE_SYMBOL_MISSING`, fail closed |
| native call returns non-zero | `DEPTH_SAFETY_NATIVE_EXECUTION_FAILED`, fail closed |
| invalid depth shape | explicit `ValueError` |
| `PLANNING_DEPTH_SAFETY_BACKEND=python` | rejected unless `PLANNING_DEPTH_SAFETY_REFERENCE=1` |
| explicit Python reference | available only for test/debug characterization |

The fail-closed seam was developed test-first.  The initial RED tests
demonstrated that missing native code and implicit Python mode were accepted;
the current GREEN tests enforce the failure policy without changing numeric
geometry.

## Fixtures and exact parity

The comparator is
`tests/compare_depth_safety_parity.py`.  It runs O and P in separate
subprocesses with the same temporary MPL fixture:

```text
MPL=/tmp/xmflight_global_route_c3/c3_mpl.npz
MPL_CONTRACT_SHA256=f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d
```

It generated no project data.  All artifacts are under
`/tmp/xmflight_depth_safety_c4/`.

Synthetic edge-case coverage: 18 named frames, each evaluated for all 105
actions.  The set includes all-far, all-near, medium, center/left/right and
top/bottom obstacles, thin vertical/horizontal stripes, image and patch
boundaries, exact/just-below/just-above threshold values, zero, NaN/Inf, and
mixed valid/invalid pixels.

Production-like coverage is deterministic synthetic input, not a claim about
real sensor data:

```text
frames=1000 (200 each: open_space, near_collision, corridor,
             asymmetric_obstacle, dense_obstacle)
production_fixture_sha256=7d392527435b96310bd02b104bb5a581a9f0b6ccdb45dd9f0ce3107a8d6a080d
combined_fixture_sha256=542d836acbc57e149cdc48e1bc599d884449489848450346975b0a7e0b67d9c1
```

The O/P result arrays were compared byte-for-byte for masks, checked/visible
patch diagnostics, capped-radius diagnostics, invalid fractions, minimum
clearance, frame counters, projection tables, and action order:

```text
MASK_BIT_PARITY=PASS
MIN_CLEARANCE_PARITY=PASS
VALID_COUNT_PARITY=PASS
EMPTY_MASK_PARITY=PASS
ACTION_ORDER_PARITY=PASS
SYNTHETIC_DEPTH_PARITY=PASS
PRODUCTION_LIKE_DEPTH_PARITY=PASS
```

## Integration parity

The same comparator also exercised public consumers:

```text
DEPTH_MASK_GENERATOR_PARITY=PASS
  100 transitions, 4 episodes, exact masks, sorted row order,
  exact offsets [0,25,50,75], exact lengths [25,25,25,25]
POLICY_MASKED_ACTION_PARITY=PASS
  100 masks, masked logits, and deterministic argmax actions
TEACHER_DEPTH_SAFETY_PARITY=PASS
  100 observations, combined masks, selected actions, and dead-end flags
```

The O generator emits its existing legacy provenance fields for the temporary
legacy fixture while P's moved generator does not.  Removing or restoring
that producer metadata is outside C4 and was deliberately not done:

```text
DEPTH_MASK_ARTIFACT_PROVENANCE=NOT_HANDLED_IN_C4
```

The mask-array and episode-mapping gate above excludes only those explicit
provenance fields; no provenance parity claim is made here.

## Benchmark

Three fresh conda subprocesses per tree used the same 1000-frame,
105-action production-like fixture, one native call per frame, and the same
MPL/config.  `resource.getrusage` supplied peak RSS.  Allocation counts were
not instrumented and are reported as such.

| Metric | O | P |
| --- | ---: | ---: |
| actions/sec mean | 259712.147181 | 272496.888020 |
| frames/sec mean | 2473.449021 | 2595.208457 |
| per-call mean ms | 0.403972 | 0.384995 |
| per-call p50 ms | 0.460651 | 0.455759 |
| per-call p95 ms | 0.475887 | 0.471527 |
| init mean s | 0.016467 | 0.016313 |
| peak RSS max KB | 150844 | 150844 |
| Python/native crossings per run | 1000 | 1000 |
| array allocations | NOT_INSTRUMENTED | NOT_INSTRUMENTED |

```text
DEPTH_SAFETY_DELTA_PERCENT=+4.922658%
P_MEETS_95_PERCENT_GATE=YES
```

The O/P native depth libraries have identical SHA256:

```text
179e165cc2ef0f47232b61c0ccdb4ad1561047aab29893b841f4f8566cf27fab
```

## Build and frozen regressions

```text
P_DEPTH_SAFETY_TARGET_BUILD=PASS
P_COLLISION_TARGET_BUILD=PASS
P_GLOBAL_ROUTE_TARGET_BUILD=PASS
P_FULL_BUILD=BLOCKED_BY_BRIDGE_PROTOCOL
```

The full P build reaches the unchanged Bridge sources and stops on their
pre-existing undeclared `ROS_WARN*`/`ROS_ERROR*` macros.  No Bridge repair was
attempted.

Frozen regression evidence remains green:

```text
COLLISION_REGRESSION=PASS
  121 numeric cases exact; 100 candidate rows exact; 100 Teacher masks exact
GLOBAL_ROUTE_REGRESSION=PASS
  127 synthetic route cases; 1000 production-like routes;
  1000 mission acceptance/order rows; 100 Teacher route/action/mask rows
```

## Scope boundary and next phase

The observation provenance producer remains outside this phase.  The native
depth owner is now characterized and gated, but the production cutover itself
is intentionally not executed in this task.

```text
NEXT_PHASE=C4.1 DEPTH SAFETY PRODUCTION CUTOVER
```

## C4.1 — Depth Safety Production Cutover (2026-08-27)

The authorized P-only cutover is complete. O remained read-only and the
production numeric seam is now formally closed over the native depth target:

~~~
C4_1_DEPTH_SAFETY_PRODUCTION_CUTOVER=PASS
FORMAL_DEPTH_SAFETY_OWNER=planning.safety.depth_safety.local_depth_action_mask
PRODUCTION_IMPLEMENTATION_COUNT=1
PYTHON_PRODUCTION_FALLBACK=NO
NATIVE_DEPTH_SAFETY_FAIL_CLOSED=PASS
DEPTH_SAFETY_OWNER_UNIQUE=YES
DEPTH_CONFIG_OWNER_UNIQUE=YES
ALGORITHM_CHANGED=NO
~~~

All P production callers reach the public seam through the runtime
environment, depth-mask worker, or policy evaluator. None owns the native
library path, ctypes handle, C ABI pointer, or buffer layout. The NumPy loop
remains an explicit reference-only path requiring
PLANNING_DEPTH_SAFETY_REFERENCE=1; no native failure can select it.

The C4 exact gates were rerun and remain green:

~~~
MASK_BIT_PARITY=PASS
DEPTH_MASK_GENERATOR_PARITY=PASS
POLICY_MASKED_ACTION_PARITY=PASS
TEACHER_DEPTH_SAFETY_PARITY=PASS
COLLISION_REGRESSION=PASS
GLOBAL_ROUTE_REGRESSION=PASS
~~~

The C4.1 three-run benchmark recorded P 270011.951375 versus O
259187.179987 actions/sec (+4.176430%), above the 95% gate. The native target
and package identity are recorded in CMake and package.xml; cmake --install
was additionally observed to stop in the conda setup.py --install-layout=deb
compatibility step before the library copy. The complete P build remains
blocked only by the unchanged Bridge/protocol ROS_WARN*/ROS_ERROR_THROTTLE
declarations.

No duplicate production implementation was deleted. Storage, diagnostics,
checkpoint/config forwarding, and the explicit Python reference remain
separate contracts. Depth-mask artifact provenance (observation_contract,
observation_source, reliable_rows, and legacy_rows) remains NOT_HANDLED and
was not changed.

Detailed caller audit, deletion ledger, build/install evidence, fixture
hashes, and remaining blockers are in
docs/DEPTH_SAFETY_PRODUCTION_CUTOVER.md.

~~~
P_DEPTH_SAFETY_TARGET_BUILD=PASS
P_FULL_BUILD=BLOCKED_BY_BRIDGE_PROTOCOL
DEPTH_MASK_ARTIFACT_PROVENANCE=NOT_HANDLED
NEXT_PHASE=C5 PYTHON SHARED VOXELMAP AND NATIVE BINDING
~~~
```
