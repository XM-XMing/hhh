# C6.0.1 — Bridge contract closure

Date: 2026-08-28

This is a P-only controlled closure. The original tree
`/home/xm/XM/xm_ws/src/planning` and the Unity checkout
`/home/xm/XM/XMflight` were read-only. No C6.1 production cutover was run.

## Port owner

The unique production owner is:

```text
P/python/planning/runtime/ports.py
  WorkerRuntimeSpec             managed workers
  DirectRuntimePortProfile      explicit direct/manual runs
```

`WorkerRuntimeSpec` is serialized with `profile=managed` and projects the
same immutable values to Unity argv, Bridge ROS arguments, and Python
endpoints. `DirectRuntimePortProfile` requires `profile=direct`; its frozen
profile preserves the Unity/O direct defaults. P Bridge has no usable
internal runtime-port defaults: missing or colliding ports fail closed before
socket initialization.

There is no separate reset port. Reset requests and reset completions use the
reliable command endpoints. Command receipts use the same command endpoints;
Unity result ACKs use the Unity result endpoint; Python result
receipts/commit ACKs use the Python result endpoint; snapshot request/response
traffic uses the two snapshot endpoints.

## Full port matrix

`O bridge default` records the old C++ constructor behavior, including its
legacy reliable-port derivation. `P bridge default` records the new
fail-closed C++ constructor. A `-1` is intentional and is not a usable
production endpoint.

| Purpose | Unity default | Unity runtime arg | O Bridge default | P Bridge default | Python default/owner | WorkerRuntimeSpec | YAML | Launch | Managed resolved value | Direct smoke value |
|---|---:|---|---:|---:|---|---|---|---|---|---|
| ROS master | no Unity port | `ROS_MASTER_URI` | environment | environment | `master_port` in `ports.py` | training `11621+i`; eval `11521` | none | none | training `11621+i`; eval `11521` | explicit `11321` |
| legacy command PUB / command input | `10253` | `-cmdSubPort` | `10253` | `-1` | `command_port_base` | `command_port` | `cmd_port: 10253` | `cmd_port=10253` | training `10553+20i`; eval `10453` | `10253` |
| legacy state telemetry | `10254` | `-statePubPort` | `10254` | `-1` | profile projection | `state_port` | `state_port: 10254` | `state_port=10254` | training `10554+20i`; eval `10454` | `10254` |
| depth telemetry | `11254` | `-depthPubPort` | `11254` | `-1` | direct profile or managed base | `depth_port` | `depth_port: 12254` | `depth_port=12254` | training `12554+20i`; eval `12454` | `11254` |
| reliable command / reset / command receipt (Python-facing) | no separate Unity default | multiplexed; no Unity argv | `-1`, derived `cmd+6` | `-1` | `python_command_port` | `python_command_port` | no field; explicit launch arg | `python_command_port=-1` unless supplied | training `10559+20i`; eval `10459` | `11259` |
| reliable command / reset / command receipt (Unity-facing) | `-1` | `-reliableCommandPort` | `-1`, derived `cmd+7` | `-1` | `unity_command_port` | `unity_command_port` | no field; explicit launch arg | `unity_command_port=-1` unless supplied | training `10560+20i`; eval `10460` | `11260` |
| Unity execution result / result ACK | `11255` | `-executionResultPort` | `-1`, derived `cmd+2` | `-1` | `unity_result_port` | `unity_result_port` | `execution_result_port: -1` | `execution_result_port=-1` | training `10555+20i`; eval `10455` | `11255` |
| Python result / receipt / commit ACK | no Unity default | no Unity argv | `-1`, derived `cmd+3` | `-1` | `python_result_port` | `python_result_port` | `python_result_port: -1` | `python_result_port=-1` | training `10556+20i`; eval `10456` | `11257` |
| Unity endpoint snapshot request/response | `11256` | `-observationSnapshotPort` | `-1`, derived `result+1` | `-1` | `unity_snapshot_port` | `unity_snapshot_port` | no YAML field | `observation_snapshot_port=-1` | training `10557+20i`; eval `10457` | `11256` |
| Python endpoint snapshot request/response | no Unity default | no Unity argv | `-1`, derived `python_result+1` | `-1` | `python_snapshot_port` | `python_snapshot_port` | `python_snapshot_port: -1` | `python_snapshot_port=-1` | training `10558+20i`; eval `10458` | `11258` |

The managed training stride is 20 for command/depth/reliable endpoints and 1
for the ROS master sequence. All ten TCP ports are validated as unique per
profile and across profiles before child startup. The 12-worker fixture
(`worker_id` 0 through 11) produced zero collisions.

## Fail-closed and ownership checks

```text
missing managed profile/field       rejected before child startup
missing direct profile               rejected before child startup
unknown or mixed profile             rejected before child startup
duplicate port within profile       rejected before child startup
duplicate port across workers       rejected before child startup
independent shell reliable offsets  absent from run_sac_guarded.sh
independent C++ reliable offsets    absent from P bridge
```

The launch chain was checked with `bash -n`; the guarded launcher now asks
the canonical Python resolver for the complete managed spec and passes the
resolved values to Bridge, Unity, and Python. Managed values are unchanged.

## Build and normal-path evidence

The O and P `unity_bridge_node` Release targets both built successfully under
`conda activate xm` and ROS Noetic. The existing isolated build directories
were used:

```text
O: /tmp/xm-cxx-c0-o-build/planning
P: /tmp/xm-cxx-c2-2-p-build
```

The target link composition is the O monolithic source versus P's split
Bridge/transport/protocol sources. O is compiled as C++14 and includes the
CUDA include path; P is compiled as C++17 without CUDA collision code. This
is a build-structure difference, not a normal transport semantic change.

```text
O binary SHA256 = 1e614e13577c1d6a3c1d46318eddd31668fd235a89e3a7456ddf6da11ddcd6e7
P binary SHA256 = 971eed793f1b45fc8b9a5d9984d6f90a8f069c3b7f3ad48c31a7d741e64e3c85
```

The same frozen Player was used for both trees:

```text
Player SHA256 = 61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365
```

## Retry and live Unity evidence

The fixed local transport fixture ran 10,000 interleaved command, receipt,
result, ACK, commit, and snapshot transactions for each binary. Each side
accepted, acknowledged, relayed, received, and committed all 10,000
transactions. Both had zero business duplicates, conflicts, pending final
state, and protocol errors. The authoritative retry counters were:

```text
O command_forward_total = 10754; command_retry_total = 754
P command_forward_total = 10766; command_retry_total = 767
```

The P retry difference is a long-tail transport effect; exactly-once
acceptance, result identity, and commit state remained exact.

| Latency (ms) | O p50 | O p95 | O p99 | O max | P p50 | P p95 | P p99 | P max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| command send → receipt | 8.342008 | 9.769074 | 10.248501 | 275.213701 | 8.340744 | 9.742707 | 10.209135 | 257.456317 |
| receipt → result send | 0.002823 | 0.005130 | 0.006901 | 0.046927 | 0.002665 | 0.004940 | 0.006286 | 0.072360 |
| result send → Unity ACK | 4.985721 | 5.099548 | 5.162466 | 5.467486 | 4.987406 | 5.100968 | 5.168201 | 5.278359 |
| total command send → commit ACK | 20.484206 | 24.280364 | 24.859736 | 290.360423 | 20.540269 | 24.349672 | 24.982962 | 272.417948 |

Twenty successful independent O runs and twenty successful independent P runs
were selected for the physical comparison. Every selected run performed a
reset completion, an exact reset snapshot lookup, and 20 reliable primitives;
all 400 primitives per side were `COMPLETE`. Every selected run had 400
result receipts, 400 Unity ACKs, and 400 commits. Unity audit
`physical_execution_total` was 400 per side. A separate O attempt had a
random dynamic-port startup bind failure before Unity launch; it is retained
as fault evidence and was not counted as a successful run.

The aggregate selected-run metrics were:

```text
O/P accepted                         400 / 400
O/P Unity command receipts           400 / 400
O/P result receipts                  400 / 400
O/P result ACKs                      400 / 400
O/P commits                          400 / 400
O/P physical executions             400 / 400
O/P business duplicates              0 / 0
O/P conflicts                        0 / 0
O/P pending final                    0 / 0
O/P protocol errors                  0 / 0
O/P command retries                 11 / 12
```

Transport artifact:
`/tmp/xmflight-c6-transport-10k-ack/comparison.json`

```text
SHA256 = cc445d171121bd58e264a6277fa56e8e406af67629a629631fd830b06660f51b
```

Real Unity artifacts:

```text
/tmp/xmflight-c6-real-unity-20/characterization.json
  SHA256 = d69737486dbf47100ee0e156103fcc226380dd08ed2d42a0e216869d09ebf307
/tmp/xmflight-c6-real-unity-recovery/characterization.json
  SHA256 = 4facb116b35f3b10d287595ea18fa792f3f442e677b20511dcd627047f8e5836
```

## ZMQ fault characterization

The fault harness exercised first `zmq_setsockopt` failure, address-in-use
bind, and first `zmq_connect` failure. No business peer was started in a
fault case. Bind failures exited with code 1 on both sides, all owned ports
were released, and ROS was cleaned up. The low-level error text differs
because O reports a combined `ZMQ connect/bind failed` while P reports the
specific `zmq_bind failed: Address already in use`.

The setsockopt fault is a real parity blocker: O ignores the first ordinary
socket-option return and continues until the harness terminates it; P reports
`zmq_setsockopt failed: Invalid argument` and exits 1. This is initialization
fault behavior only; normal socket types, HWM, linger, timeout, poll, retry,
thread ownership, and message order were not changed.

```text
ZMQ_OPTION_FAILURE_PARITY = FAIL
ZMQ_BIND_FAILURE_PARITY   = PASS (same fail-closed exit/cleanup class)
ZMQ_CONNECT_FAILURE       = functional fail-closed on both; wording differs
poll fault                = not controllable in this harness
FAULT_PROCESS_CLEANUP     = PASS
```

Fault artifact:
`/tmp/xmflight-c6-zmq-faults/fault_characterization.json`

```text
SHA256 = 75fffae0e6569b29b47ed2750898f91e124a98eaa13536755b889d98d24fb858
```

## Tests

```text
tests/test_c6_0_1_port_contract.py
tests/test_c6_bridge_protocol_transport.py
tests/test_reliable_v4_worker_runtime_spec.py
tests/test_reliable_v4_runtime_instance_lifecycle.py
tests/test_unity_bridge_result_runtime_integration.py
tests/test_reliable_v4_awac_training_contract.py
```

Results:

```text
focused unit set:                 33 passed, 25 deselected
P Bridge ROS regression:           24 passed
O Bridge ROS regression:           24 passed
Python compileall:                 PASS
guarded launcher bash syntax:      PASS
```

The first ROS attempt was an environment-only failure caused by replacing
ROS's absolute Python path with `PYTHONPATH=python`; rerunning with
`/opt/ros/noetic/lib/python3/dist-packages:/home/xm/XM/src/python` passed both
O and P suites. No test or runtime failure was hidden by counting that first
attempt as a pass.

## P files touched in this controlled slice

```text
python/planning/runtime/ports.py
python/planning/runtime/worker.py
python/planning/runtime/collection_worker.py
python/planning/runtime/supervisor.py
python/planning/diagnostics/reliable_single_worker.py
python/planning/teacher/collection_config.py
python/planning/rl/train_sac.py
scripts/evaluate_policy_unity_managed.sh
scripts/run_sac_guarded.sh
src/bridge/bridge_node.cpp
tests/test_c6_0_1_port_contract.py
tests/test_c6_bridge_protocol_transport.py
tests/test_unity_bridge_result_runtime_integration.py
tests/test_unity_runtime_result_consumer.py
tests/test_reliable_v4_awac_training_contract.py
docs/BRIDGE_CONTRACT_CLOSURE.md
docs/BRIDGE_PROTOCOL_TRANSPORT_PARITY.md
docs/CXX_FOUNDATION_MIGRATION.md
docs/PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md
```

Only P files in this list were modified. O, Unity, BC/AWAC mathematics,
formal data, and the temporary harnesses under `/tmp` were not part of the
source migration.

## Closure decision

```text
C6_0_1_BRIDGE_CONTRACT_CLOSURE = FAIL
PORT_OWNER = python/planning/runtime/ports.py::WorkerRuntimeSpec + DirectRuntimePortProfile
PORT_OWNER_UNIQUE = YES
MANAGED_PORT_PARITY = PASS
DIRECT_DEFAULT_PARITY = PASS (explicit profile; P internal defaults intentionally absent)
TWELVE_WORKER_PORT_COLLISION_COUNT = 0
TRANSPORT_TRANSACTIONS = 10000 per O/P
REAL_UNITY_PRIMITIVES = O=400, P=400
RETRY_BEHAVIOR = ACCEPTABLE_TRANSPORT_RECOVERY
DUPLICATE_PHYSICAL_EXECUTION_COUNT = 0
COMMAND_CONFLICT_COUNT = 0
PENDING_FINAL_COUNT = 0
PROTOCOL_ERROR_COUNT = 0
ZMQ_OPTION_FAILURE_PARITY = FAIL
ZMQ_BIND_FAILURE_PARITY = PASS
FAULT_PROCESS_CLEANUP = PASS
P_SPLIT_BRIDGE_PRODUCTION_CANDIDATE = NO
NORMAL_RUNTIME_BEHAVIOR_CHANGED = NO
GPU_RUNTIME_VALIDATION_DEFERRED = YES
ALGORITHM_CHANGED = NO
NEXT_PHASE = C6.0.1 CONTINUE INVESTIGATION
```

C6.1 production cutover is not authorized by this result and was not
executed. The remaining blocker is a deliberate decision about whether to
match O's ignored noncritical `zmq_setsockopt` failure or to keep P's stricter
fail-closed behavior and formally accept that fault-contract change.

## C6.0.2 — ZMQ socket-option criticality contract

C6.0.2 resolved the former coarse option-failure gate by classifying every
production `zmq_setsockopt` call in P. The complete call-order inventory,
option values, O/P behavior matrix, injection harnesses, and artifact hashes
are in [`ZMQ_SOCKET_OPTION_CONTRACT.md`](ZMQ_SOCKET_OPTION_CONTRACT.md).

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
REQUIRED_FAULT_FAIL_CLOSED = PASS
OPTIONAL_DEGRADED_MODE = PASS
FAULT_PROCESS_CLEANUP = PASS
FAULT_PORT_RELEASE = PASS
NORMAL_TRANSPORT_REGRESSION = PASS
NORMAL_UNITY_RUNTIME_REGRESSION = PASS
P_SPLIT_BRIDGE_PRODUCTION_CANDIDATE = YES
NORMAL_RUNTIME_BEHAVIOR_CHANGED = NO
GPU_RUNTIME_VALIDATION_DEFERRED = YES
ALGORITHM_CHANGED = NO
NEXT_PHASE = C6.1 BRIDGE PRODUCTION CUTOVER
```

The five option classes are `ZMQ_LINGER` (lifecycle cleanup),
`ZMQ_ROUTER_HANDOVER` (exactly-once), `ZMQ_SUBSCRIBE` (correctness), and the
optional performance controls `ZMQ_SNDHWM`/`ZMQ_RCVHWM`. Required and unknown
faults fail closed after transport/context cleanup; optional faults emit a
degraded metric and warning while preserving normal message behavior. The
normal 10,000-transaction and 400-primitive Unity regressions remain exact.
C6.1 was not executed.

## C6.1 — Production cutover result (2026-08-28)

The authorized cutover is complete in P.  The split Bridge is the only formal
P implementation; O and Unity remained read-only.  No old P flat source was
present for deletion, no O file was changed, and compatibility forwarding
headers were retained as public/test references.  The detailed deletion
ledger is in [`BRIDGE_PRODUCTION_CUTOVER.md`](BRIDGE_PRODUCTION_CUTOVER.md).

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
NEXT_PHASE=C7 FINAL C++ BUILD AND FREEZE
```

Cutover evidence includes exact canonical payloads, 10,000 transport
transactions per binary, 20 Unity runs per binary with 400 primitives each,
all five socket-option fault rows, startup/shutdown cleanup, and the native
compute regressions.  The O test-source exact-dictionary mismatch is retained
as read-only baseline evidence: six assertions fail because O emits its own
diagnostic fields; normalized O/P business parity and P's corresponding
24-test runtime suite pass.  C7 was not started.
