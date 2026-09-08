# C6.1 — Bridge production cutover

Date: 2026-08-28

This is the P-only production cutover record.  The original tree
`/home/xm/XM/xm_ws/src/planning` (O) was read-only, the Unity checkout was
frozen, and no formal data, training run, or commit was created.  Commands
were run with `conda activate xm`.

## Cutover result

P's verified split Bridge is now the only formal P implementation behind the
unchanged production executable name `unity_bridge_node`:

```text
BridgeNode = planning::bridge::UnityBridgeNode
  CommandGateway   = src/bridge/command_gateway.cpp
  ResultGateway    = src/bridge/result_gateway.cpp
  SnapshotGateway  = src/bridge/snapshot_gateway.cpp
  ResetGateway     = src/bridge/reset_gateway.cpp
  TelemetryBridge  = src/bridge/telemetry_bridge.cpp
  Protocol         = src/protocol/*
  Transport        = src/transport/*
  executable       = src/unity_bridge_main.cpp
```

Each audited behavior has one P definition.  The old O monolith remains the
read-only behavior baseline; it was not copied, edited, or deleted.

```text
FORMAL_BRIDGE_IMPLEMENTATION_COUNT=1
OLD_BRIDGE_FORMAL_CALLER_COUNT=0
MISSING_BEHAVIOR_COUNT=0
DUPLICATE_BEHAVIOR_COUNT=0
```

## Inventory and deletion ledger

| Item | O baseline | P classification | Formal callers before/after | Replacement/evidence | Action |
| --- | --- | --- | --- | --- | --- |
| `src/unity_bridge_node.cpp` | One historical monolithic owner | `SUPERSEDED` in O; absent in P | P `0/0` | `src/unity_bridge_main.cpp` plus `src/bridge/*` | No P deletion; O untouched |
| Gateway implementations | Methods live in the O monolith | `SAFE_CANDIDATE` split owners | One definition per gateway in P | C6.1 owner-definition scan and focused tests | No duplicate P source found |
| Protocol codecs | O monolith plus root protocol header | Canonical split protocol | One definition per wire owner | golden bytes, hash, and codec comparators | No duplicate P codec found |
| Transport/context | O monolith | `SAFE_CANDIDATE` `BridgeTransport`/`ZmqSocket` | One P context/socket owner | RAII close and 10,000-transaction fixture | No duplicate P transport found |
| Root protocol/broker headers | O public headers | `TEST_REFERENCE`/`PUBLIC_COMPAT` | Existing references remain | Thin forwarding headers only; no runtime definitions | Retained |
| Old O binary/path references | O build/runtime baseline | `TEST_REFERENCE` only where historical evidence is documented | P formal callers `0` | static formal-wiring scan | No O file or path deleted |

The P tree contained no old flat implementation eligible for deletion at
cutover time.  Compatibility wrappers were retained because they are public
or test references and contain only canonical forwarding includes.  No item
was deleted from O.

## Build and target closure

The single P target remains `unity_bridge_node`, with C++17, ROS, ZeroMQ,
MessagePack, and OpenSSL dependencies.  Its source list is the manifest
`PLANNING_BRIDGE_SOURCES` in `CMakeLists.txt`; CMake has one executable target
and one install entry.  No optimization or runtime tuning flag was changed
for C6.1.

```text
P_FULL_BUILD=PASS
BRIDGE_COMPILE=PASS
P flags=-O3 -DNDEBUG -O2 -Wall -Wextra -std=c++17
P LTO=NONE
P architecture flags=NONE
P OpenMP compile flags=NONE (Bridge target)
P visibility/exception/RTTI changes=NONE
```

The O comparison build was retained as the C++14 monolith with its existing
CUDA include path.  This structural/toolchain difference is not presented as
a protocol or algorithm change.

## Runtime ownership and wiring

Formal launch and runtime wiring resolves ports through
`python/planning/runtime/ports.py` → `WorkerRuntimeSpec` /
`DirectRuntimePortProfile`.  Bridge consumes resolved ports and rejects
missing required values before socket initialization.  Launchers and formal
runtime callers resolve the P binary, while the external contract remains:

```text
binary       = unity_bridge_node
ROS node     = unity_bridge_node
endpoints    = command, receipt, result/ACK, snapshot, reset, telemetry
owner        = BridgeNode / split
schema       = 4
source list  = python/planning/runtime/bridge_identity.py
```

The P startup record includes binary path, binary SHA-256, protocol schema,
resolved ports, and `runtime_instance_id`.  The source-manifest SHA is
`27fc44226fe033a0dd63bcfb1fbf1326022684f9b06e479d1af7ab08e40acf9e`.

## Parity and fault evidence

Canonical command, receipt, result, ACK, snapshot request/response, reset,
and telemetry business fields were byte-compared.  Independent process
nonce/time and retry-shaped diagnostics were excluded only by the comparator;
command hashes, execution association, frame counts, outcomes, collision
values, depth identity, and snapshot hashes were not excluded.

```text
PROTOCOL_PARITY=PASS
COMMAND_GATEWAY_PARITY=PASS
RESULT_GATEWAY_PARITY=PASS
SNAPSHOT_GATEWAY_PARITY=PASS
RESET_GATEWAY_PARITY=PASS
TELEMETRY_PARITY=PASS
```

The corrected transport fixture completed 10,000 transactions for each
binary.  O and P each reported 10,000 accepted/receipt/ACK/commit/snapshot
transactions and zero business duplicates, conflicts, pending final state,
or protocol errors.  The frozen Player completed 20 independent runs per
binary with reset plus 20 reliable primitives per run: O=400 and P=400
physical executions, all accepted, with no duplicate, conflict, pending, or
protocol-error outcome.

The five option classes and startup faults were rerun against P:

```text
ZMQ_LINGER          required   fail-closed/nonzero/released   PASS
ZMQ_ROUTER_HANDOVER required   fail-closed/nonzero/released   PASS
ZMQ_SUBSCRIBE       required   fail-closed/nonzero/released   PASS
ZMQ_SNDHWM          optional   warning/degraded/continued    PASS
ZMQ_RCVHWM          optional   warning/degraded/continued    PASS
UNKNOWN_OPTION_COUNT=0
bind/address-in-use/partial-startup/shutdown cleanup=PASS
FAULT_PROCESS_CLEANUP=PASS
```

O's ignored ordinary `zmq_setsockopt` failure remains historical baseline
behavior.  The cutover records P's authorized canonical fail-closed policy;
it does not claim the O behavior is normally incorrect.

## Frozen Unity and native compute gates

The actual Player used was
`/home/xm/XM/xm_ws/src/unity/XMflight.x86_64`.  Its executable SHA-256 was
`61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365`; the
adjacent loaded `Assembly-CSharp.dll` SHA-256 was
`7ffb7bab78d84b45980b17daabae8872b8ba772af2821aa9f8a7da4c54a6dabc`.
Reset, 25-frame primitive execution, ACK/commit, endpoint snapshot, depth,
collision, and physics checks passed for both trees.

The post-cutover native gates were:

```text
COLLISION_REGRESSION   = PASS (121 exact; candidate accept 100; Teacher masks 100 x 105)
GLOBAL_ROUTE_REGRESSION= PASS (127 synthetic; 1000 production-like; mission/order 1000; Teacher 100)
DEPTH_SAFETY_REGRESSION= PASS (synthetic; production-like; generator; policy; Teacher mask)
```

No collision CUDA/hybrid capability claim was made.  GPU collision runtime
validation remains deferred/not available and was not used by the cutover.

## Test and artifact record

Relevant checks passed with temporary byte/cache artifacts under `/tmp`:

```text
compileall                                      PASS
C6.1 static/owner/port/option contract tests    24 passed, 1 deselected
broader protocol/gateway/runtime suite         126 passed, 30 deselected
P real-process Bridge integration               24 passed
O/P canonical payload fixture comparator        PASS
O/P transport comparator                        PASS
O/P frozen Unity characterization               PASS
```

The O legacy ROS test file has six exact-dictionary assertion failures because
the O executable itself emits the existing diagnostic metrics fields; its
other 18 tests passed.  This read-only test mismatch was not used to fail the
P cutover: the normalized O/P runtime comparator and the frozen Unity
characterization passed, while the O test source was not modified.

Key runtime artifacts:

```text
/tmp/xmflight-c6-1-real-unity-final/characterization.json
  SHA256=74d64fa1c87cdbd1d70b34630aa7f03459ed0df10b0dbabafc0ea53e01c509ae
/tmp/xmflight-c6-1-option-faults/option_fault_matrix.json
  SHA256=e6aee02b3c10fe542c7951bbdeb69a3fe11315df89f6c12ce40384e5de0df93c
/tmp/xmflight-c6-1-fault-characterization/fault_characterization.json
  SHA256=f06840e0bdd5fbdccfb89798ee4c66ce8d57ca635627831254a2fb66c3b2d0fa
/tmp/xmflight-c6-1-transport-o-retry/comparison.json
  SHA256=57feb51e9fcd490bc626a8a0f67278fa7ad92ae1485ed28c6751b41062f55c4f
/tmp/xmflight-c6-1-transport-p-retry/comparison.json
  SHA256=ae24529573e9f3948fa6457fd212c3ab32454ddde37c86bd9258853eca1136d8
```

## Final gate

```text
C6_1_BRIDGE_PRODUCTION_CUTOVER=PASS
FORMAL_BRIDGE_OWNER=planning::bridge::UnityBridgeNode (BridgeNode)
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
BRIDGE_BINARY_SHA256=63b00f6a0b31aa72ace3e1c437477e7ee8b670934386f3bb1d645745c302d9ed
RUNTIME_ASSEMBLY_SHA256=7ffb7bab78d84b45980b17daabae8872b8ba772af2821aa9f8a7da4c54a6dabc
UNITY_PLAYER_SHA256=61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365
GPU_RUNTIME_VALIDATION_DEFERRED=YES
ALGORITHM_CHANGED=NO
```

No C7 work was started.  The next authorized phase is `C7 FINAL C++ BUILD
AND FREEZE`.
