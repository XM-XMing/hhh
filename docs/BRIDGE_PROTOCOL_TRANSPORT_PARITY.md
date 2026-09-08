# C6 Bridge, Protocol, and Transport parity

Date: 2026-08-28

~~~text
O=/home/xm/XM/xm_ws/src/planning        (read-only behavior baseline)
P=/home/xm/XM/src                      (candidate and only modified tree)
UNITY=/home/xm/XM/XMflight             (read-only frozen Player)
C6_BRIDGE_PROTOCOL_TRANSPORT=PARTIAL
COMPILE_REPAIR_SCOPE_EXPANDED=NO
ALGORITHM_CHANGED=NO
~~~

This is a controlled C6 audit. It does not cut over the Bridge, change the
wire schema, change collision/A*/depth behavior, or modify Unity. No formal
data was collected and no commit was created.

## Inventory and ownership classification

| Module | O owner | P owner | Classification | C6 finding |
| --- | --- | --- | --- | --- |
| COMMON | root include/planning headers, ROS nodes, shared message types | compatibility headers plus split include/planning/{bridge,geometry,navigation,protocol,transport} and the same ROS nodes | EXACT_MOVE, STRUCTURAL_REFACTOR, SAFE_CANDIDATE | Root compatibility headers are forwarding aliases; no duplicate runtime implementation. |
| VOXEL_MAP | private collision geometry | geometry/voxel_map.*, planning_voxel_map | NEW, SAFE_CANDIDATE | Existing C1 immutable shared owner was not changed in C6. |
| COLLISION | collision_checker.cpp and CUDA collision_checker_cuda.cu | geometry/collision_checker.* | STRUCTURAL_REFACTOR, REMOVED (pre-C6 CUDA counterpart), BLOCKED | No collision source was changed. CUDA/hybrid remains outside this C6 scope and is not claimed incorrect. |
| DEPTH_SAFETY | depth_safety.cpp | geometry/depth_safety.* | EXACT_MOVE, STRUCTURAL_REFACTOR, SAFE_CANDIDATE | No depth source was changed. |
| GLOBAL_ROUTE | no O C++ owner | navigation/global_route_planner.* | NEW, BLOCKED | No route source or native A* behavior was changed. |
| BRIDGE | monolithic src/unity_bridge_node.cpp | unity_bridge_main.cpp and bridge/* | STRUCTURAL_REFACTOR, SAFE_CANDIDATE | One P source owner per target; normal/reliable paths pass. Production candidate remains gated by direct-port drift and fault-path parity. |
| PROTOCOL | wire/broker code in the monolith and root protocol header | protocol/* plus root forwarding headers | EXACT_MOVE, STRUCTURAL_REFACTOR, SAFE_CANDIDATE | Constants, canonical encoders, hashes, and tested message contracts match. |
| TRANSPORT | socket/context ownership in the monolith | transport/*, BridgeTransport, ZmqSocket | STRUCTURAL_REFACTOR, NEW, SAFE_CANDIDATE | One context/socket owner and matching normal bind/connect topology. Option-failure fault injection remains untested. |

MISSING_BEHAVIOR_COUNT=0 and DUPLICATE_BEHAVIOR_COUNT=0 refer to the audited
normal Bridge/protocol/transport surface. The pre-C6 absence of the O CUDA
target is retained as a separate blocked/out-of-formal-scope item; it was not
deleted during C6.

## Compile repair and build baseline

The first P full-build log reproduced only declaration seams:

- result_gateway.cpp and snapshot_gateway.cpp used ROS logging macros without
  the ROS logging declaration include.
- telemetry_wire.cpp used the O-local AsInt32 helper without defining it in
  the split translation unit.

The only P source repairs were:

~~~text
src/bridge/result_gateway.cpp       + #include <ros/ros.h>
src/bridge/snapshot_gateway.cpp     + #include <ros/ros.h>
src/protocol/telemetry_wire.cpp     + exact local AsInt32 helper from O
~~~

These restore declarations/types only. They add no branch, field, socket,
thread, retry, or business-logic behavior; therefore
COMPILE_REPAIR_SCOPE_EXPANDED=NO.

Both builds used conda activate xm and ROS Noetic:

~~~text
O: cmake --build /tmp/xm-cxx-c0-o-build/planning -j2
P: cmake --build /tmp/xm-cxx-c2-2-p-build -j2
~~~

Both exited zero. The O target compiled the monolithic Bridge plus its
CUDA/collision targets. The P target compiled generated messages,
planning_voxel_map, planning_collision_checker, planning_global_route,
planning_depth_safety, voxel_map_parity_fixture, the split
unity_bridge_node, visualization_node, and map_loader_node.

The Bridge compile flags were otherwise comparable: both used -O3 -DNDEBUG
-O2 -Wall -Wextra, with O -std=c++14 and P -std=c++17. Neither Bridge target
used LTO, explicit architecture flags, OpenMP flags, special visibility
flags, or disabled RTTI/exceptions. O's include path also contains
/usr/local/cuda-11.8/include; P has no CUDA include or link input. P links
the fifteen split Bridge/protocol/transport objects; O links the single
src/unity_bridge_node.cpp.o. The only P warning is the existing unused
AsInt32 in endpoint_observation_snapshot_wire.cpp.

Build evidence:

~~~text
/tmp/xmflight-c6-o-full-build.log
/tmp/xmflight-c6-p-full-build.log
/tmp/xmflight-c6-p-bridge-red.log       (pre-repair red reproduction)

O unity_bridge_node SHA256 = 1e614e13577c1d6a3c1d46318eddd31668fd235a89e3a7456ddf6da11ddcd6e7
P unity_bridge_node SHA256 = c864f02be31bbc87364e2495e5a92b3f763246f611c7452664a1d5721b2c8070
~~~

The differing executable hashes are expected from C++ standard, source
layout, and generated code differences; they are not used as a behavior
parity condition.

CMakeLists.txt and package.xml were not modified in C6.  The pre-existing P
package export metadata and split CMake target/install layout remain part of
the earlier C2-C5 migration record; C6 only validated the resulting full
build and did not widen package/install scope.

## Protocol parity

The canonical P include/planning/protocol/xm_protocol.hpp and O's root
include/planning/xm_protocol.hpp are byte-identical:

~~~text
SHA256=e45566e4bec13fb8e1bb4d0e6e7b0162b83cffdec5019aefb447fc66df04d6e3
~~~

All 27 language-neutral fixture files under tests/fixtures have identical O/P
bytes. Representative fixture hashes are:

~~~text
observation_ref.msgpack.hex = 33e28f0fb6b53de43ba01e012ab160895424ef9db9910299ddb76c9712bff6d3
snapshot.msgpack.hex         = 91b22577a1e0c83bf8a4240b7657090be1b6233e27e435a0801b872e2b171cfd
snapshot_request.msgpack.hex = efe9fba8cb15277b83679224b2a43aa8f81bef4f4d342dccb09104c21545ca47
complete.result_msgpack.hex  = 998c01a474c8f95a44f0e27df1040d9db01e2e80a3cee9759d9c44e75eeaf6e4
complete.result_payload_hash = 6905efd66ea66729d6875a303361f9f7e2a4f5c635061d033f334434ef10cfd2
command_sequence_hash        = 055322190be0d59e9ba778a16bd07a2d84c805416b0be9f921762e3095b2928f
~~~

O/P standalone C++ contracts produced no normalized output difference for
primitive execution v4, endpoint snapshots, reset, result/observation
brokers, and receipts. Python golden vectors and C++ contracts cover valid,
duplicate, conflict, malformed, complete, failed, rejected, ACK, receipt,
snapshot request/response, reset, and protocol-error classifications.

~~~text
PROTOCOL_CONSTANT_PARITY=PASS
CANONICAL_BYTES_PARITY=PASS
HASH_PARITY=PASS
WIRE_CODEC_PARITY=PASS
~~~

## Ownership and transport topology

planning::transport::BridgeTransport is the only P owner of the ZMQ context
and socket lifetimes. ZmqSocket closes each socket by RAII;
BridgeTransport::Close tears down snapshot, result, telemetry, and command
transports before terminating the context. BridgeNode owns gateway lifetime
and destruction order. Gateways own brokers, counters, ROS callbacks, and
protocol decisions; they do not create a second ZMQ context. Reliable adapter
maps route identities but do not own socket contexts.

The normal endpoint mapping is:

| Endpoint | Managed/default mapping |
| --- | --- |
| command/state/depth | command, command+1, configured managed depth |
| Unity result / Python result | command+2 / command+3 |
| Unity snapshot / Python snapshot | command+4 / command+5 |
| Python command / Unity command | command+6 / command+7 |
| managed config depth | O/P 12254 |
| direct Bridge constructor depth default | O/P 11254 |

Managed launch/config parity is exact. Direct-default parity is not: a direct
invocation of either Bridge uses 11254, while formal managed config uses
12254. This known 11254 vs 12254 integration risk was intentionally not
changed in C6.

Normal socket roles/options were audited: command PUB plus Python/Unity
ROUTER endpoints, telemetry SUB endpoints, result ROUTER endpoints, snapshot
ROUTER endpoints, HWM/LINGER settings, handover settings, and inproc result
monitor ownership are present in the corresponding P transport classes.
Reconnect/handover/stale recovery, duplicate/conflict handling, and late
packet rejection are covered by the existing real-process integration suite.

A remaining static failure-path difference is recorded rather than hidden:
P's ZmqSocket checks some option-setter failures and closes on failure, where O
historically ignores several zmq_setsockopt return values. No fault-injection
test was run, so this is not promoted to exact fault-path parity; C6.1 must
characterize it before production cutover.

## Gateway and runtime evidence

The P real-process Bridge integration suite passed 24 passed. The O/P
transport-only fixture sent 1000 interleaved result transactions through four
Unity identities plus one Python identity, with one intentional malformed
packet. Both sides reported the exact normalized metrics:

~~~json
{"accepted":1000,"duplicate":0,"conflict":0,"pending":0,"pending_receipt":0,"commit":1000,"ack":1000,"python_relay":1000,"receipt":1000,"duplicate_receipt":0,"protocol_error":1}
~~~

The complete metrics file SHA was
83c43fcda04c1c6e45b8ae54cc1db04fb254b163406d60be7aa080a714377f86 for both
O and P stress runs. This establishes zero cross-talk, zero residual pending
state, and only the intentional malformed-packet protocol error.

The same frozen Player was run once against O and once against P for 20
reliable primitives. The Player SHA was
61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365. Both runs
passed reset, 20/20 primitive execution, 25/25 frame application, result
ACK/commit, endpoint snapshot, depth dimensions/SHA, collision flag, and
physics state-value checks. With identical explicit 60-second retry windows,
O/P metrics were byte-identical (SHA
ca5535fac012a749c937787611164849bc0c27b3750b0ac1e98ad40f47d16ebb).
Execution IDs, result status/reason, frame counts, command hashes, endpoint
physics values, and depth SHA matched. state_id and simulation time are
runtime-generated and differed by launch timing; each endpoint reference was
internally exact and identity-valid.

The default 0.5-second live fixture also passed all business checks, but one
P run observed one extra command retry/forward while O observed zero. This
timing-sensitive counter difference is retained as a C6.1 characterization
blocker; it is not labeled a semantic failure because final ACK/commit and
identity checks remained exact.

~~~text
TRANSPORT_ONLY_PARITY=PASS
COMMAND_GATEWAY_PARITY=PASS       (functional and equal-budget counter path)
RESULT_GATEWAY_PARITY=PASS
SNAPSHOT_GATEWAY_PARITY=PASS
RESET_GATEWAY_PARITY=PASS          (wire/contract and live reset path)
TELEMETRY_PARITY=PASS              (source/codec and live state/depth checks)
REAL_UNITY_RUNTIME_PARITY=PASS
PHYSICS_PARITY=PASS
DEPTH_PARITY=PASS
COLLISION_PARITY=PASS
RUNTIME_IDENTITY_PARITY=PASS
PROCESS_CLEANUP=PASS
~~~

## Tests and files changed in C6

~~~text
compileall: PASS
focused protocol/gateway/transport suite: 99 passed, 1 deselected
P real-process Bridge integration: 24 passed
O/P transport-only stress: PASS / PASS
O/P frozen Player smoke: 20 / 20 primitives, all checks PASS
~~~

P-side C6 changes are limited to compile declarations, test compatibility,
the characterization test, and documentation:

~~~text
src/bridge/result_gateway.cpp
src/bridge/snapshot_gateway.cpp
src/protocol/telemetry_wire.cpp
tests/test_primitive_execution_protocol_contract.py
tests/test_unity_bridge_result_runtime_integration.py
tests/test_primitive_execution_v4_csharp_contract.py
tests/test_primitive_reset_v4_csharp_contract.py
tests/test_c6_bridge_protocol_transport.py
docs/BRIDGE_PROTOCOL_TRANSPORT_PARITY.md
docs/CXX_FOUNDATION_MIGRATION.md
docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md
~~~

The C# test edits only point at the current read-only Unity canonical
Runtime/Protocol files. They do not alter Unity. Runtime artifacts are under
/tmp/xmflight-c6-*; no project data was generated.

## Gate and next phase

~~~text
P_FULL_BUILD=PASS
BRIDGE_COMPILE=PASS
MANAGED_PORT_PARITY=PASS
DIRECT_DEFAULT_PARITY=FAIL
P_SPLIT_BRIDGE_PRODUCTION_CANDIDATE=NO
GPU_RUNTIME_VALIDATION_DEFERRED=YES
ALGORITHM_CHANGED=NO
NEXT_PHASE=C6.1 BRIDGE PRODUCTION CUTOVER
~~~

C6.1 is not executed here. It must first resolve or explicitly freeze the
managed/direct depth-port authority, characterize the default retry-counter
difference and ZMQ option-failure path, then repeat the normal and frozen
Player gates before any Bridge cutover. C7 is not started.

## C6.0.1 — Bridge contract closure (2026-08-28)

The follow-up closure made `python/planning/runtime/ports.py` the unique
production port owner. `WorkerRuntimeSpec` owns managed training/evaluation
topology and `DirectRuntimePortProfile` owns explicit direct/manual topology.
All ten runtime TCP ports are validated before child startup; reset and
receipts are multiplexed on the command/result endpoints and do not introduce
additional ports. Managed values are unchanged. The direct frozen profile is
an explicit profile matching Unity/O direct defaults, while P's C++ defaults
remain `-1` by design so missing arguments fail closed.

The complete matrix, exact managed/direct values, tests, artifacts, and fault
evidence are in `docs/BRIDGE_CONTRACT_CLOSURE.md`.

~~~text
C6_0_1_BRIDGE_CONTRACT_CLOSURE=FAIL
PORT_OWNER=python/planning/runtime/ports.py::WorkerRuntimeSpec+DirectRuntimePortProfile
PORT_OWNER_UNIQUE=YES
MANAGED_PORT_PARITY=PASS
DIRECT_DEFAULT_PARITY=PASS (explicit profile)
TWELVE_WORKER_PORT_COLLISION_COUNT=0
TRANSPORT_TRANSACTIONS=10000 per O/P
REAL_UNITY_PRIMITIVES=O=400,P=400
RETRY_BEHAVIOR=ACCEPTABLE_TRANSPORT_RECOVERY
DUPLICATE_PHYSICAL_EXECUTION_COUNT=0
COMMAND_CONFLICT_COUNT=0
PENDING_FINAL_COUNT=0
PROTOCOL_ERROR_COUNT=0
ZMQ_OPTION_FAILURE_PARITY=FAIL
ZMQ_BIND_FAILURE_PARITY=PASS
FAULT_PROCESS_CLEANUP=PASS
P_SPLIT_BRIDGE_PRODUCTION_CANDIDATE=NO
NORMAL_RUNTIME_BEHAVIOR_CHANGED=NO
GPU_RUNTIME_VALIDATION_DEFERRED=YES
ALGORITHM_CHANGED=NO
NEXT_PHASE=C6.0.1 CONTINUE INVESTIGATION
~~~

The fixed 10,000-transaction comparator reported O/P command retry totals of
754/767, all accepted/receipt/ACK/commit/snapshot counts 10000/10000, and
zero business duplicates, conflicts, pending final state, or protocol errors.
The result-to-Unity-ACK and command-to-commit-ACK percentiles are recorded in
the closure document. Twenty successful independent O runs and twenty P runs
each produced 400 physical reliable primitives. One additional O startup
attempt failed at a dynamically allocated bind before Unity launch and was
retained as fault evidence, not counted as a successful run.

ZMQ fault injection found equivalent fail-closed bind cleanup, but different
first-`zmq_setsockopt` behavior: O continued because the ordinary option
return was ignored, whereas P exited 1 with a specific error. This is the
remaining C6.0.1 blocker; C6.1 was not run.

## C6.0.2 — ZMQ socket-option criticality contract

The C6.0.2 option audit is complete. P now routes all production socket-option
calls through one criticality contract: three required classes fail closed,
two optional HWM classes record degraded mode and continue, and unknown
options fail closed. See [`ZMQ_SOCKET_OPTION_CONTRACT.md`](ZMQ_SOCKET_OPTION_CONTRACT.md)
for the complete O/P startup inventory and per-option fault evidence.

```text
C6_0_2_ZMQ_OPTION_CONTRACT = PASS
PRODUCTION_SOCKET_OPTION_COUNT = 5
REQUIRED_OPTION_COUNT = 3
OPTIONAL_OPTION_COUNT = 2
UNKNOWN_OPTION_COUNT = 0
ZMQ_OPTION_FAILURE_CONTRACT = PASS
O_MATCHES_CANONICAL = MIXED
P_MATCHES_CANONICAL = YES
O_HISTORICAL_FAULT_POLICY_DEFECT = YES
FAULT_PROCESS_CLEANUP = PASS
FAULT_PORT_RELEASE = PASS
NORMAL_TRANSPORT_REGRESSION = PASS
NORMAL_UNITY_RUNTIME_REGRESSION = PASS
P_SPLIT_BRIDGE_PRODUCTION_CANDIDATE = YES
NORMAL_RUNTIME_BEHAVIOR_CHANGED = NO
NEXT_PHASE = C6.1 BRIDGE PRODUCTION CUTOVER
```

The O first-option continuation remains a historical policy defect under the
canonical contract; it is not a normal-runtime parity failure. Required,
optional, unknown, normal 10,000-transaction, and Unity 400-primitive gates
were exercised on P. C6.1 was not executed.

## C6.1 — Bridge production cutover (2026-08-28)

C6.1 is now complete in P.  O remained read-only and the Unity Player
remained frozen.  The split `planning::bridge::UnityBridgeNode` is the single
formal owner behind the unchanged `unity_bridge_node` executable.  The
canonical owner map is `BridgeNode -> CommandGateway, ResultGateway,
SnapshotGateway, ResetGateway, TelemetryBridge, Protocol, Transport`; the
source-level owner scan found one definition per audited behavior and no P
flat `src/unity_bridge_node.cpp` implementation or formal caller.

```text
C6_1_BRIDGE_PRODUCTION_CUTOVER=PASS
FORMAL_BRIDGE_IMPLEMENTATION_COUNT=1
OLD_BRIDGE_FORMAL_CALLER_COUNT=0
MISSING_BEHAVIOR_COUNT=0
DUPLICATE_BEHAVIOR_COUNT=0
P_FULL_BUILD=PASS
BRIDGE_COMPILE=PASS
PROTOCOL_PARITY=PASS
COMMAND_GATEWAY_PARITY=PASS
RESULT_GATEWAY_PARITY=PASS
SNAPSHOT_GATEWAY_PARITY=PASS
RESET_GATEWAY_PARITY=PASS
TELEMETRY_PARITY=PASS
PORT_CONTRACT=PASS
ZMQ_OPTION_CONTRACT=PASS
FAULT_PROCESS_CLEANUP=PASS
REAL_UNITY_RUNTIME_PARITY=PASS
PHYSICS_PARITY=PASS
DEPTH_PARITY=PASS
COLLISION_PARITY=PASS
COLLISION_REGRESSION=PASS
GLOBAL_ROUTE_REGRESSION=PASS
DEPTH_SAFETY_REGRESSION=PASS
DUPLICATE_PHYSICAL_EXECUTION_COUNT=0
COMMAND_CONFLICT_COUNT=0
PENDING_FINAL_COUNT=0
PROTOCOL_ERROR_COUNT=0
GPU_RUNTIME_VALIDATION_DEFERRED=YES
ALGORITHM_CHANGED=NO
NEXT_PHASE=C7 FINAL C++ BUILD AND FREEZE
```

The corrected O/P transport fixture completed 10,000 transactions per
binary.  The frozen Player completed 20 independent runs per binary with 20
reliable primitives per run, for O=400 and P=400 physical executions.  Both
normalized results had exact business accounting and zero duplicate,
conflict, pending, or protocol-error outcomes.  The five C6.0.2 socket-option
fault classes, bind/address-in-use, partial startup, shutdown cleanup, and
the native Collision, Global Route, and Depth Safety gates all passed.

The P Bridge binary SHA-256 is
`63b00f6a0b31aa72ace3e1c437477e7ee8b670934386f3bb1d645745c302d9ed`; the
source-manifest SHA-256 is
`27fc44226fe033a0dd63bcfb1fbf1326022684f9b06e479d1af7ab08e40acf9e`.
The full deletion/retention ledger is in
[`BRIDGE_PRODUCTION_CUTOVER.md`](BRIDGE_PRODUCTION_CUTOVER.md).

The O legacy integration test still has six exact-metrics assertion failures
against fields emitted by O's own executable; its 18 other tests passed.  The
cutover gate uses the normalized O/P business-field comparator and the
successful frozen Unity characterization, and no O test or source was
modified.  C7 was not started.
