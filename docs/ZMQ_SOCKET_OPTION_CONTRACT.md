# C6.0.2 — ZMQ socket-option criticality contract

Date: 2026-08-28

This is a P-only fault-policy migration. The original tree
`/home/xm/XM/xm_ws/src/planning` and the Unity checkout
`/home/xm/XM/XMflight` remained read-only. Normal socket values, socket types,
retry timing, polling, thread count, send/receive order, and shutdown order
were not changed. C6.1 was not executed.

## Decision

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

The candidate decision is about P's split Bridge only. It does not claim CUDA
collision parity; GPU collision validation remains deferred and unavailable.

## Real startup option inventory

The inventory was taken from the actual O/P C++ Bridge initialization paths.
No `SNDTIMEO`, `RCVTIMEO`, `IMMEDIATE`, `RECONNECT_IVL`, `TCP_KEEPALIVE`, or
`IDENTITY` startup setter exists in either C++ Bridge. O/P
`ZMQ_ROUTER_MANDATORY` and `ZMQ_SNDHWM` reads in the result relay are runtime
`getsockopt` diagnostics, not startup setters. `zmq_socket_monitor` is also
not a socket option.

### O call order

Owner: `src/unity_bridge_node.cpp::UnityBridgeNode::Init`.

```text
 1 cmd_pub_socket/PUB                  ZMQ_LINGER=0
 2 state_sub_socket/SUB                ZMQ_LINGER=0
 3 depth_sub_socket/SUB                ZMQ_LINGER=0
 4 python_command_router_socket/ROUTER ZMQ_LINGER=0
 5 unity_command_router_socket/ROUTER  ZMQ_LINGER=0
 6 result_router_socket/ROUTER         ZMQ_LINGER=0
 7 observation_snapshot_router/ROUTER  ZMQ_LINGER=0
 8 python_result_router_socket/ROUTER  ZMQ_LINGER=0
 9 python_snapshot_router_socket/ROUTER ZMQ_LINGER=0
10 cmd_pub_socket/PUB                  ZMQ_SNDHWM=send_hwm_ (4)
11 state_sub_socket/SUB                ZMQ_RCVHWM=recv_hwm_ (4)
12 depth_sub_socket/SUB                ZMQ_RCVHWM=recv_hwm_ (4)
13 python_command_router_socket/ROUTER ZMQ_RCVHWM=recv_hwm_ (4)
14 python_command_router_socket/ROUTER ZMQ_SNDHWM=send_hwm_ (4)
15 unity_command_router_socket/ROUTER  ZMQ_RCVHWM=recv_hwm_ (4)
16 unity_command_router_socket/ROUTER  ZMQ_SNDHWM=send_hwm_ (4)
17 result_router_socket/ROUTER         ZMQ_RCVHWM=recv_hwm_ (4)
18 result_router_socket/ROUTER         ZMQ_SNDHWM=send_hwm_ (4)
19 observation_snapshot_router/ROUTER  ZMQ_RCVHWM=recv_hwm_ (4)
20 observation_snapshot_router/ROUTER  ZMQ_SNDHWM=send_hwm_ (4)
21 python_result_router_socket/ROUTER  ZMQ_RCVHWM=recv_hwm_ (4)
22 python_result_router_socket/ROUTER  ZMQ_SNDHWM=send_hwm_ (4)
23 python_snapshot_router_socket/ROUTER ZMQ_RCVHWM=recv_hwm_ (4)
24 python_snapshot_router_socket/ROUTER ZMQ_SNDHWM=send_hwm_ (4)
25 python_result_router_socket/ROUTER  ZMQ_ROUTER_HANDOVER=1
26 observation_snapshot_router/ROUTER  ZMQ_ROUTER_HANDOVER=1
27 python_command_router_socket/ROUTER ZMQ_ROUTER_HANDOVER=1
28 unity_command_router_socket/ROUTER  ZMQ_ROUTER_HANDOVER=1
29 state_sub_socket/SUB                ZMQ_SUBSCRIBE="", size=0
30 depth_sub_socket/SUB                ZMQ_SUBSCRIBE="", size=0
```

O ignores return values for calls 1–24 and 29–30. Calls 25–28 are checked;
the first failed call returns from `Init`, and the destructor later closes the
already-created resources.

### P call order

The option policy owner is
`src/transport/zmq_option_contract.cpp` plus
`src/transport/zmq_socket.cpp`. Transport owners supply the socket purpose.

```text
 1 command_pub/PUB                  ZMQ_LINGER=0
 2 command_pub/PUB                  ZMQ_SNDHWM=send_hwm (4)
 3 python_command_router/ROUTER    ZMQ_LINGER=0
 4 python_command_router/ROUTER    ZMQ_RCVHWM=receive_hwm (4)
 5 python_command_router/ROUTER    ZMQ_SNDHWM=send_hwm (4)
 6 unity_command_router/ROUTER     ZMQ_LINGER=0
 7 unity_command_router/ROUTER     ZMQ_RCVHWM=receive_hwm (4)
 8 unity_command_router/ROUTER     ZMQ_SNDHWM=send_hwm (4)
 9 python_command_router/ROUTER    ZMQ_ROUTER_HANDOVER=1
10 unity_command_router/ROUTER     ZMQ_ROUTER_HANDOVER=1
11 state_sub/SUB                   ZMQ_LINGER=0
12 state_sub/SUB                   ZMQ_RCVHWM=receive_hwm (4)
13 state_sub/SUB                   ZMQ_SUBSCRIBE="", size=0
14 depth_sub/SUB                   ZMQ_LINGER=0
15 depth_sub/SUB                   ZMQ_RCVHWM=receive_hwm (4)
16 depth_sub/SUB                   ZMQ_SUBSCRIBE="", size=0
17 result_router/ROUTER            ZMQ_LINGER=0
18 result_router/ROUTER            ZMQ_RCVHWM=receive_hwm (4)
19 result_router/ROUTER            ZMQ_SNDHWM=send_hwm (4)
20 python_result_router/ROUTER     ZMQ_LINGER=0
21 python_result_router/ROUTER     ZMQ_RCVHWM=receive_hwm (4)
22 python_result_router/ROUTER     ZMQ_SNDHWM=send_hwm (4)
23 python_result_router/ROUTER     ZMQ_ROUTER_HANDOVER=1
24 observation_snapshot_router/ROUTER ZMQ_LINGER=0
25 observation_snapshot_router/ROUTER ZMQ_RCVHWM=receive_hwm (4)
26 observation_snapshot_router/ROUTER ZMQ_SNDHWM=send_hwm (4)
27 python_snapshot_router/ROUTER  ZMQ_LINGER=0
28 python_snapshot_router/ROUTER  ZMQ_RCVHWM=receive_hwm (4)
29 python_snapshot_router/ROUTER  ZMQ_SNDHWM=send_hwm (4)
30 observation_snapshot_router/ROUTER ZMQ_ROUTER_HANDOVER=1
```

All P calls go through `ZmqSocket::SetInt` or `SetBytes`; required and
unknown failures return `false`, causing each transport and the shared ZMQ
context to close. Optional failures return success only after recording the
degraded socket/option identity.

## Criticality classification

| Option | Criticality | Evidence-based reason | P degraded behavior |
|---|---|---|---|
| `ZMQ_LINGER=0` | `REQUIRED_FOR_LIFECYCLE_CLEANUP` | Bridge owns nine sockets and must not retain pending frames during teardown; the contract requires bounded close. | fail closed |
| `ZMQ_SNDHWM=4` | `OPTIONAL_PERFORMANCE` | HWM bounds queues and changes backpressure/performance. Existing nonblocking send plus retry/commit accounting preserves the business contract when one HWM setter is unavailable. | warning + `degraded_socket_options`, continue |
| `ZMQ_RCVHWM=4` | `OPTIONAL_PERFORMANCE` | Same queue-bound classification; the receiver can continue with the library default while exact application identities and commit hashes remain validated. | warning + `degraded_socket_options`, continue |
| `ZMQ_ROUTER_HANDOVER=1` | `REQUIRED_FOR_EXACTLY_ONCE` | Stable runtime identities reconnect through ROUTER sockets. Handover is the boundary that lets a replacement pipe become the registered peer without routing the durable result to a stale pipe. | fail closed |
| `ZMQ_SUBSCRIBE=""` | `REQUIRED_FOR_CORRECTNESS` | The empty subscription is what admits all state/depth telemetry frames; without it the observation path is silent. | fail closed |

Any option not in this table is `UNKNOWN` and defaults to fail closed. No
unknown production option was found.

## Canonical fault policy

```text
REQUIRED_FOR_CORRECTNESS       -> error, nonzero exit, close all resources
REQUIRED_FOR_EXACTLY_ONCE      -> error, nonzero exit, close all resources
REQUIRED_FOR_LIFECYCLE_CLEANUP -> error, nonzero exit, close all resources
OPTIONAL_PERFORMANCE           -> warning, record degraded option, continue
OPTIONAL_RECOVERY_TUNING       -> warning, record degraded option, continue
UNKNOWN                        -> error, nonzero exit, close all resources
```

The optional path was allowed to continue only after the runtime gates below
passed. The degraded field is diagnostic only; no wire message, command order,
retry interval, HWM value, or result schema changed.

## Per-option fault matrix

The final matrix used a preload shim that failed the first call matching the
numeric option, restricted to the `unity_bridge_node` executable. Each row
started a fresh O/P Bridge with a fresh 10-port direct profile. `-15` means
the harness observed the Bridge alive past the 2.5 second startup window and
then terminated it; `1` means the Bridge rejected initialization.

| Option | Criticality | O result | P result | Ports released | P canonical | O canonical |
|---|---|---|---|---|---|---|
| `ZMQ_LINGER` | required cleanup | `rc=-15`, stayed alive; ignored | `rc=1`, detailed error | O/P 10/10 | YES | NO |
| `ZMQ_SNDHWM` | optional performance | `rc=-15`, stayed alive; no warning | `rc=-15`, warning `command_pub/ZMQ_SNDHWM` | O/P 10/10 | YES | NO: no degraded record |
| `ZMQ_RCVHWM` | optional performance | `rc=-15`, stayed alive; no warning | `rc=-15`, warning `python_command_router/ZMQ_RCVHWM` | O/P 10/10 | YES | NO: no degraded record |
| `ZMQ_ROUTER_HANDOVER` | required exactly once | `rc=1`, checked error | `rc=1`, detailed error | O/P 10/10 | YES | YES |
| `ZMQ_SUBSCRIBE` | required correctness | `rc=-15`, stayed alive; ignored | `rc=1`, detailed error | O/P 10/10 | YES | NO |

The startup fault fixture created no business peers, so accepted business
requests were zero by construction. Required P faults never reached Bridge
ready and therefore accepted zero requests. The optional path was separately
run with business peers below.

### Cleanup evidence

For every option/P required-fault row, the Bridge exited or was terminated,
all ten owned TCP ports were immediately bindable again, and the dedicated
roscore exited with code zero. No Bridge/roscore process remained after each
case. This is process-level evidence for socket, ZMQ context, ROS, and worker
thread teardown; these internal objects have no separate production counter.
The P `BridgeTransport::Close` path closes snapshot, result, telemetry, and
command socket owners before `zmq_ctx_term`.

The final startup matrix artifact is:

```text
/tmp/xmflight-c6-0-2-option-faults-final/option_fault_matrix.json
SHA256=e8a91dee2745893368ee367370f218c1996c03b7508f427cacafe9ec6d8a483a
```

## Optional degraded-mode runtime gate

The final P binary was fault-injected once for each optional option. Each
fixture completed 1,000 transport transactions with accepted/ACK/commit /
snapshot/hash counts all 1,000 and duplicate/conflict/pending/protocol-error
counters all zero. Each also completed a frozen Unity reset plus 20 reliable
primitives: 20/20 `COMPLETE`, 20/20 durable ACK, 20/20 committed, 20 physical
executions, zero duplicate/conflict/pending/protocol-error outcomes.

| Fault | Diagnostic field | 1,000-transaction artifact | 20-primitive artifact |
|---|---|---|---|
| `ZMQ_SNDHWM` | `command_pub/ZMQ_SNDHWM` | `/tmp/xmflight-c6-0-2-degraded-sndhwm-final/P/bridge_result_metrics.json`<br>`SHA256=06a78d9dfe2cfb505bd4d6147c25a93f48b6ba358eab08b0bfa42477cc4891df` | `/tmp/xmflight-c6-0-2-degraded-sndhwm-unity-final/characterization.json`<br>`SHA256=c29c7034f7be7621cf73133e9bcd9dbca75d35941c86391601a44e36d5a28036` |
| `ZMQ_RCVHWM` | `python_command_router/ZMQ_RCVHWM` | `/tmp/xmflight-c6-0-2-degraded-rcvhwm-final/P/bridge_result_metrics.json`<br>`SHA256=55c7e771c6fa855fe70bcac81de3b94ea3e51942798deb7069a86fe6a60c8ce1` | `/tmp/xmflight-c6-0-2-degraded-rcvhwm-unity-final/characterization.json`<br>`SHA256=9e6421fd4985fa909aa3a91e62f739527c66139e3ef002c36e016d717f0aea85` |

## Required and unknown tests

New P tests:

```text
tests/test_c6_0_2_zmq_socket_option_contract.py
tests/cpp/zmq_socket_option_contract.cpp
```

The focused unit set passed `19 passed, 1 deselected`; the new C++ unknown
option harness compiled and reported `C++ ZMQ option contract tests: 2
passed`. P and O existing Bridge ROS integration each passed `24 passed`.
Python compileall passed with a temporary cache prefix.

The first preload attempt also touched the Python test process and produced a
Python-side `Function not implemented`; that harness run was discarded and
the corrected shim was restricted to the Bridge executable. One normal P
10k attempt hit a transient `zmq_bind: Address already in use` before any
transaction; it was retained as a failed startup attempt and excluded from
the successful normal fixture.

## Normal-path regression

The final P executable completed the same 10,000-transaction fixture as O:

```text
O accepted/receipt/commit/ack/snapshot = 10000/10000/10000/10000/10000
P accepted/receipt/commit/ack/snapshot = 10000/10000/10000/10000/10000
O retries = 774
P retries = 757
O command p95 = 9.737831 ms; total p95 = 24.302728 ms
P command p95 = 9.791825 ms; total p95 = 24.349588 ms
O/P duplicate = 0; conflict = 0; pending = 0; protocol_error = 0
```

Final P metric artifact:

```text
/tmp/xmflight-c6-0-2-normal-10k-p-final-retry/P/bridge_result_metrics.json
SHA256=5d4098dfeb698a1d71456bcd910a53624de7e2960808fa0ffa50182da29f8aed
```

The frozen Player was run in 20 independent sessions per side, each with a
reset and 20 reliable primitives:

```text
O = 20/20 sessions, 400/400 physical executions, all COMPLETE
P = 20/20 sessions, 400/400 physical executions, all COMPLETE
O/P duplicate = 0; conflict = 0; pending = 0; protocol_error = 0
```

```text
Player SHA256=61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365
P normal Unity artifact SHA256=50f5a5edaa3ab9fdd1131b11b48bd662529b8732e54bc9fe8ac4bd81209526ec
```

The final P executable used for the last build and fault matrix is:

```text
SHA256=5f10e3f1033675100d0fcee3529aacd9343af2502beee3d699acee0a18cf0eea
```

No BC/AWAC, Collision, Global Route, Depth Safety, Python algorithm, Unity,
formal data, or normal transport parameter was changed.

## C6.1 cutover closure (2026-08-28)

The P split Bridge was cut over as the sole formal implementation after the
socket-option contract was rerun.  O remained the read-only baseline.  P
returned nonzero and released all ten ports for required `ZMQ_LINGER`,
`ZMQ_ROUTER_HANDOVER`, and `ZMQ_SUBSCRIBE` faults.  P continued with a
warning/degraded record and released all ports for optional `ZMQ_SNDHWM` and
`ZMQ_RCVHWM` faults.  Unknown option count remained zero.  Bind,
address-in-use, partial-startup, and shutdown cleanup also passed.

```text
C6_1_BRIDGE_PRODUCTION_CUTOVER=PASS
ZMQ_OPTION_CONTRACT=PASS
PRODUCTION_SOCKET_OPTION_COUNT=5
REQUIRED_OPTION_COUNT=3
OPTIONAL_OPTION_COUNT=2
UNKNOWN_OPTION_COUNT=0
FAULT_PROCESS_CLEANUP=PASS
FAULT_PORT_RELEASE=PASS
NORMAL_TRANSPORT_REGRESSION=PASS
NORMAL_UNITY_RUNTIME_REGRESSION=PASS
P_SPLIT_BRIDGE_PRODUCTION_CANDIDATE=YES
GPU_RUNTIME_VALIDATION_DEFERRED=YES
ALGORITHM_CHANGED=NO
NEXT_PHASE=C7 FINAL C++ BUILD AND FREEZE
```

O's first-option continuation is recorded as historical fault-policy behavior,
not as proof of incorrect normal behavior.  The cutover therefore preserves
P's authorized fail-closed policy and does not claim CUDA collision parity.
Full per-row return codes, release counts, and the deletion ledger are in
[`BRIDGE_PRODUCTION_CUTOVER.md`](BRIDGE_PRODUCTION_CUTOVER.md).
