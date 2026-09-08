# Primitive Execution Schema v4

状态：P0-K 设计文档；P0-L1/P0-L2a/P0-L2b 的纯模型、canonical encoding 与 Unity 侧纯 lifecycle seam 已实现。本文不改变当前 v3 runtime，也不授权修改 bridge、ROS message 或 Python transition 执行逻辑。

关联 RED：episode=145、step=22、execution_id=4435486932233309751。该 RED 的物理首丢帧是 UNITY_PUB_TO_BRIDGE：Unity 的 25 个 frame 均已 apply，frame 24 的本地发送调用返回成功，但 bridge 没有收到 frame 24。

## 1. 设计目标和非目标

### 1.1 目标

v4 将 primitive 的“是否执行完成”和“frame telemetry 是否完整到达”分成两个接口：

1. FRAME_APPLIED 是可丢失的诊断/时序 telemetry。
2. PrimitiveExecutionResult 是 primitive transaction 的事实来源。
3. result 通过带 ACK、重传和去重的可靠 channel 传输。
4. 一个 execution_id 最多产生一个 Python transition commit。
5. COMPLETE 必须绑定真实 endpoint observation，不能用 result 到达后的第一条 state 模糊替代。
6. v3/v4 不得静默互通；schema mismatch 必须 fail-fast。

### 1.2 非目标

本阶段不处理：

- SAC、AWAC、BC、reward、replay 或 RL 超参数；
- HWM 调优、arbitrary sleep、COMPLETE 多发、primitive retry；
- 放宽 25/25 的真实 apply invariant；
- 删除现有 execution transport audit；
- 把 PUB/SUB 改成可靠 channel 的临时补丁。

## 2. 术语和所有权

| 术语 | 定义 |
|---|---|
| physical primitive | Unity 实际执行一次的固定 primitive；execution_id 唯一标识它 |
| frame telemetry | 对每个 applied physics frame 发布的 FRAME_APPLIED 记录；允许丢失 |
| final result | Unity 对一次 physical primitive 产生的不可变终态结果 |
| result retransmission | 重发完全相同的 final result payload；不重新执行 action |
| transition commit | Python 将一次合法执行结果和 endpoint observation 写入环境 transition 的单次操作 |
| duplicate result | 同一 (runtime_instance_id, execution_id) 且 payload identity 相同的再次到达 |
| protocol error | schema、execution、command identity 或 endpoint binding 不一致；必须 fail-closed |

所有权：

- Unity 拥有 physical execution、applied-frame 计数和 final result 生成。
- bridge 拥有 reliable result relay、pending result store 和 Unity ACK。
- Python 拥有 endpoint observation binding、deduplication 和 transition commit。
- PUB/SUB telemetry 不再拥有 primitive success 判定。

## 3. v4 envelope 和 schema version

v4 transaction message 使用显式 map/envelope，而不是依赖 v3 positional array 的隐式字段位置。所有 v4 message 必须包含：

~~~text
schema_version: uint32 = 4
message_type: string
runtime_instance_id: string
execution_id: uint64
~~~

runtime_instance_id 必须包含 worker/runtime 唯一身份；在 12-worker 场景中，dedup key 使用 (runtime_instance_id, execution_id)，不能只依赖本地 execution number。

建议 canonical identity：

~~~text
command_sequence_hash = SHA-256(canonical MessagePack(command frames in index order))
result_payload_hash   = SHA-256(canonical MessagePack(PrimitiveExecutionResult
                                                       without result_payload_hash))
~~~

canonical encoding、整数宽度、浮点 NaN/Inf 拒绝规则和字段排序固定见第 3.1 节；不同实现产生的 hash 不得互相兼容猜测。

## 3.1 P0-L2a canonical encoding appendix

This appendix fixes the identity encoding used by the P0-L2a golden vectors.
It does not change transaction semantics or define a transport.

### 3.1.1 MessagePack container rules

- Every canonical map key is UTF-8 text and keys are emitted in ascending
  lexicographic order by UTF-8 bytes.
- Map and array headers use the shortest MessagePack container header that can
  represent the count: fixmap/fixarray, then map16/array16, then
  map32/array32.
- Every text value uses `str8` for byte length <=255, `str16` for <=65535, and
  `str32` otherwise. Fixstr is not used.
- `null` is MessagePack `nil` (`0xc0`). Boolean values, if added to a future
  field, use MessagePack `true`/`false`; no current v4 identity field is
  boolean.
- `bytes32` is MessagePack `bin8`, length 32, followed by exactly 32 bytes.
- Arrays preserve semantic order. NaN, positive infinity, and negative
  infinity are rejected before encoding.

### 3.1.2 Fixed scalar widths

| Semantic field type | MessagePack representation |
|---|---|
| `uint32` | `uint32` (`0xce`) followed by 4 big-endian bytes |
| `uint64` | `uint64` (`0xcf`) followed by 8 big-endian bytes |
| `int32` | `int32` (`0xd2`) followed by 4 big-endian bytes |
| `int64` | `int64` (`0xd3`) followed by 8 big-endian bytes |
| `float32` | `float32` (`0xca`) followed by 4 IEEE-754 big-endian bytes |
| UTF-8 string | `str8`/`str16`/`str32` according to byte length |
| `bytes32` | `bin8`, length 32 |
| absent endpoint field | `nil` |

The result map contains 15 pairs, in this exact UTF-8 key order:

~~~text
applied_frame_count
command_sequence_hash
endpoint_observation_ref
endpoint_sim_time_ns
endpoint_state_id
execution_id
first_applied_state_id
last_applied_frame_index
message_type
reason_code
requested_frame_count
result_generation
runtime_instance_id
schema_version
status
~~~

`schema_version`, `requested_frame_count`, `applied_frame_count`, and
`result_generation` are `uint32`; `execution_id` and
`endpoint_sim_time_ns` are `uint64`; `first_applied_state_id` and
`endpoint_state_id` are `int64`; `last_applied_frame_index` is `int32`.
`command_sequence_hash` is `bytes32`, decoded from its semantic SHA-256 hex
form. `result_payload_hash` is excluded from this map before hashing.
`result_generation` is `uint32(0)` for the initial result and every
retransmission.

`endpoint_observation_ref`, when present, is the complete P0-O1
`ObservationRef`: a seven-pair map ordered as `depth_id`, `episode_id`,
`reset_id`, `runtime_instance_id`, `schema_version`, `sim_time_ns`, `state_id`.
`schema_version` is `uint32`; `state_id` is `int64`; `sim_time_ns` is `uint64`.
Its runtime identity must equal the enclosing result runtime identity, and its
state/time must equal the enclosing COMPLETE endpoint fields. `depth_id` is the
authoritative endpoint depth identity; `episode_id` and `reset_id` are never
inferred by a bridge or consumer.

### 3.1.3 Command sequence identity

The command identity is SHA-256 over one array of frames in submitted order.
Each frame is a three-pair map ordered as `action`, `command_id`,
`frame_index`. `action` is an ordered array of `float32`, `command_id` is
`int64`, and `frame_index` is `uint32`. A reordered frame, changed action
value, changed frame count, or non-finite action value therefore cannot retain
the original command identity.

Both hashes are SHA-256 over the exact canonical bytes above. Implementations
must reject a mismatch; they must not normalize received bytes or fall back to
v3 identity rules.

## 3.2 P0-L2b Unity pure result lifecycle appendix

This appendix defines the pure Unity-side lifecycle seam. It does not define a
socket, broker, bridge relay, Python transition commit, or PUB/SUB telemetry
change.

### 3.2.1 State and resource boundary

The lifecycle has these states:

~~~text
IDLE -> EXECUTING -> FINAL_RESULT_PENDING_ACK -> ACKED
                         ^                         |
                         +------ Poll retry -------+
~~~

One lifecycle instance permits at most one active execution and one pending
immutable final result. A second `BeginExecution` while the first execution is
`EXECUTING` or `FINAL_RESULT_PENDING_ACK` is a protocol error; it cannot
replace the first identity or pending payload. After a matching ACK, the same
instance may begin the next sequential execution. No unbounded pending table is
introduced.

The counters have distinct meanings:

- physical execution count increments only when `BeginExecution` accepts a new
  execution;
- result generation count increments once when the terminal result is
  canonicalized and cached;
- send attempt count includes the initial send and each eligible retry;
- `result_generation` remains `uint32(0)` for the initial result and every
  retransmission.

### 3.2.2 Immutable result delivery

`PrimitiveExecutionResultLifecycle` caches the exact canonical result bytes and
their SHA-256 `result_payload_hash`. `Poll(now_ms)` returns no result before the
retry interval and at most one retry per call when the deadline is reached. A
large clock jump does not create a burst of retries. Every retry uses the same
execution identity, command hash, result hash, status, and serialized bytes; it
never invokes physical execution again.

`COMPLETE` is generated only when the v4 result invariants hold: requested 25,
applied 25, last frame 24, and endpoint state equal to first state plus 24.
An incomplete `COMPLETE` is rejected before result generation. `REJECTED` is a
pre-execution terminal result with applied count 0 and last frame -1.
`CANCELLED` and `FAILED` preserve the actual applied prefix.

### 3.2.3 PrimitiveExecutionResultAck

The pure ACK model is:

~~~text
schema_version: uint32 = 4
message_type: "PrimitiveExecutionResultAck"
runtime_instance_id: string
execution_id: uint64
ack_status: "DURABLE_RECEIVED"
result_payload_hash: bytes32
command_sequence_hash: bytes32
~~~

Its canonical MessagePack map has seven pairs, in UTF-8 key order:

~~~text
ack_status
command_sequence_hash
execution_id
message_type
result_payload_hash
runtime_instance_id
schema_version
~~~

ACK acceptance requires all identity fields, schema, message type, and
`DURABLE_RECEIVED` to match. A wrong ACK is rejected and leaves the pending
result intact. A matching ACK clears pending and enters `ACKED`; the same ACK
received again is an idempotent duplicate outcome and cannot execute or
generate anything.

### 3.2.4 PrimitiveExecutionResultReceiptAck

The bridge-to-Python relay has a separate application receipt. `zmq_send`
success is only a local socket enqueue result and is not Python application
delivery. Python sends this message after validating the complete result
identity and before transition/replay commit:

~~~text
schema_version: uint32 = 4
message_type: "PrimitiveExecutionResultReceiptAck"
runtime_instance_id: string
execution_id: uint64
receipt_status: "RECEIVED"
result_payload_hash: bytes32
command_sequence_hash: bytes32
~~~

Its canonical MessagePack map has seven pairs, in UTF-8 key order:

~~~text
command_sequence_hash
execution_id
message_type
receipt_status
result_payload_hash
runtime_instance_id
schema_version
~~~

The bridge accepts the receipt only when every identity field and hash matches
the stored immutable result. A matching first receipt clears
`RESULT_SENT_PENDING_RECEIPT`; a repeated identical receipt is a duplicate and
has no side effect. A conflicting receipt is a protocol error and keeps the
result receipt-pending. Until the first matching receipt, reconnect/retry sends
the exact same serialized result bytes. Receipt acknowledgement does not itself
commit a transition and does not authorize a second transition.

## 4. Frame telemetry

### 4.1 FRAME_APPLIED

FRAME_APPLIED 至少包含：

~~~text
schema_version
runtime_instance_id
execution_id
frame_index                 # 0..requested_frame_count-1
frame_count                 # requested frame count
command_id
state_id
sim_time_ns
execution_status             # FRAME_APPLIED or COMPLETE telemetry marker
~~~

Unity audit 中的 frame_applied、serialization_success、state_publish_attempted 和本地 try_send_return 继续保留。它们用于 debug、gap detection 和 timing，不是 transaction success receipt。

允许：

- frame telemetry 丢失；
- bridge 或 planning 没有收到 frame 24；
- audit 记录显示 UNITY_PUB_TO_BRIDGE。

不允许：

- 用缺失 telemetry 推断 Unity 没有 apply；
- 用 telemetry 的完整性代替 final result；
- 用 telemetry 的缺失把已经有合法 result 的 primitive 判为失败。

## 5. PrimitiveExecutionResult

### 5.1 字段

| 字段 | 类型 | 必须 | 约束 |
|---|---|---:|---|
| schema_version | uint32 | 是 | 必须为 4 |
| message_type | string | 是 | PrimitiveExecutionResult |
| runtime_instance_id | string | 是 | 与 runtime manifest 和 channel identity 一致 |
| execution_id | uint64 | 是 | 一次 physical primitive 的唯一 ID |
| status | enum | 是 | COMPLETE、REJECTED、FAILED、CANCELLED |
| requested_frame_count | uint32 | 是 | 当前 deterministic primitive 必须为 25 |
| applied_frame_count | uint32 | 是 | 真实 apply 数，不是 bridge receipt 数 |
| first_applied_state_id | int64 | 条件 | applied_frame_count > 0 时必须存在 |
| endpoint_state_id | int64 | 条件 | COMPLETE 必须存在 |
| last_applied_frame_index | int32 | 是 | 无 apply 为 -1，否则为最后真实 apply index |
| reason_code | enum | 是 | COMPLETE 为 NONE，其他 status 必须给出原因 |
| command_sequence_hash | bytes32 | 是 | 绑定提交给 Unity 的完整 ordered command sequence |
| endpoint_sim_time_ns | uint64 | 条件 | COMPLETE 必须存在 |
| endpoint_observation_ref | ObservationRef map | 条件 | COMPLETE 必须携带完整 ObservationRef，精确取得 s_{t+1} |
| result_payload_hash | bytes32 | 是 | ACK/dedup identity；重传不得变化 |
| result_generation | uint32 | 是 | 初次生成固定为 0；重传不递增 |

result_generation 不是重试计数。result retransmission 必须复用同一个 generation 和同一 payload。

### 5.2 COMPLETE invariant

只有以下条件全部满足时，Unity 才能生成 status=COMPLETE：

~~~text
requested_frame_count     == 25
applied_frame_count       == 25
last_applied_frame_index  == 24
endpoint_state_id         == first_applied_state_id + 24
endpoint_sim_time_ns      is present
command_sequence_hash     matches the submitted primitive
~~~

Python 仍必须重新验证这些 invariant；不能信任 status=COMPLETE 字符串本身。任何不满足都必须是 protocol error 或 FAILED，不能 commit transition。

### 5.3 其他 status

#### REJECTED

primitive 尚未开始执行。必须立即生成 result，不允许 Debug.LogWarning + return 后让 Python 等 timeout。

~~~text
applied_frame_count      == 0
last_applied_frame_index == -1
first_applied_state_id   == null
endpoint_state_id        == null
~~~

典型 reason_code：

~~~text
OVERLAPPING_EXECUTION
MALFORMED_COMMAND
CTRL_LATENCY_NOT_ZERO
INVALID_FRAME_COUNT
INVALID_COMMAND
SCHEMA_MISMATCH
~~~

#### CANCELLED

primitive 已开始，但被外部事件主动中断。必须保留真实 applied prefix。

典型 reason_code：STOP、RESET、EXTERNAL_OVERRIDE、SAFETY_STOP。

CANCELLED 不得写成正常 Replay transition，也不得被 Python 猜成 timeout。

#### FAILED

primitive 执行过程中发生 primitive/runtime 自身失败。必须保留真实 applied prefix。

典型 reason_code：COLLISION、INTERNAL_ERROR、INVALID_RUNTIME_STATE。

FAILED 不得伪装成 COMPLETE，也不得由 Python 通过“收到了多少 frame”猜测。

当物理 primitive 已真实完成 25 个 frame、但权威 endpoint observation
capture 失败时，允许唯一的显式失败形态：

~~~text
status                    == FAILED
reason_code               == PHYSICS_COMPLETE_OBSERVATION_FAILED
requested_frame_count     == 25
applied_frame_count       == 25
last_applied_frame_index  == 24
first_applied_state_id    is present
endpoint_state_id         == null
endpoint_sim_time_ns      == null
endpoint_observation_ref  == null
~~~

这不是 COMPLETE，也不能提交 transition/replay。该扩展只放宽
`FAILED` 的 status-specific invariant；字段、canonical map 和 hash 规则不变。

## 6. Reliable result 和 ACK

### 6.1 逻辑消息

可靠 channel 至少承载：

~~~text
PrimitiveExecutionResult
PrimitiveExecutionResultAck
PrimitiveExecutionResultReceiptAck
~~~

PrimitiveExecutionResultAck 至少包含：

~~~text
schema_version
message_type = PrimitiveExecutionResultAck
runtime_instance_id
execution_id
ack_status                    # DURABLE_RECEIVED / COMMITTED / DUPLICATE
result_payload_hash
command_sequence_hash
~~~

ACK 的 hash 必须与 result 相同；ACK 不能只带 execution_id 而不验证 payload identity。

### 6.2 pending 和 result retransmission

Unity 状态机：

~~~text
IDLE
  -> EXECUTING
  -> FINAL_RESULT_PENDING_ACK
  -> ACKED
~~~

规则：

1. primitive 完成后生成一次 immutable result。
2. result 写入 Unity pending table。
3. Unity 发送 COMPLETE/REJECTED/CANCELLED/FAILED result。
4. 未收到匹配 ACK 时，按配置的 retry interval 重发完全相同 payload。
5. ACK 超时不能触发 action、primitive 或 frame 重执行。
6. 收到同一 execution 的 DURABLE_RECEIVED 后，Unity 可以删除本地 pending；bridge 仍需保留 result，直到 planning 侧完成或明确 terminal outcome。
7. 连接断开后，pending result 进入 reconnect queue；恢复连接后用同一 execution/hash 继续发送。

DURABLE_RECEIVED 表示 bridge 已把不可变 result 放入 bounded pending store，不表示 Python 已经新增 transition。bridge 到 Python 的 relay 必须继续可靠投递；Python commit 后 bridge 记录 COMMITTED。

### 6.3 duplicate / idempotency

Python execution ledger 以 (runtime_instance_id, execution_id) 为主键，并保存：

~~~text
status
result_payload_hash
command_sequence_hash
transition_commit_id
endpoint_state_id
~~~

重复到达时：

| 条件 | 行为 |
|---|---|
| 相同 execution + 相同 result hash + 已 committed | 返回 DUPLICATE ACK；不产生新 transition/reward/replay/step |
| 相同 execution + 相同 hash + pending | 不重复执行；继续原 commit 流程 |
| 相同 execution + 不同 result hash | PROTOCOL_ERROR，停止该 runtime |
| 不同 runtime identity 但相同 execution_id | 不得合并；按 runtime identity 隔离 |

目标语义：

~~~text
at-least-once reliable result delivery
+ execution_id/result-hash deduplication
=> exactly-once transition accounting
~~~

## 7. Reliable transport 设计候选

### 7.1 首选：ZeroMQ DEALER/ROUTER

这是 P0-L 的设计候选，不是当前实现。

每个 isolated runtime 使用独立 resolved ports：

~~~text
Unity DEALER  --connect-->  bridge ROUTER  (result ingress)
Python DEALER --connect-->  bridge ROUTER  (result relay/commit ACK)
~~~

bridge 作为 result broker：

- ingress ROUTER 接收 Unity result；
- 按 (runtime_instance_id, execution_id) 去重；
- 将 immutable result 放入 bounded pending store；
- 向 Unity 返回 DURABLE_RECEIVED；
- 向 Python relay result，直到收到匹配的 RESULT_RECEIVED receipt；
- Python 的 COMMITTED/DUPLICATE 再回传 bridge；
- bridge 记录最终结果，并在 Unity ACK 重传时返回相同 identity。

这不是把 COMPLETE 放进另一个 PUB socket。PUB/SUB 继续只承载 state/depth/frame telemetry。

### 7.2 必须固定的 transport contract

runtime config/manifest 必须记录：

~~~text
result_ingress_bind_host
result_ingress_port
result_relay_bind_host
result_relay_port
unity_connect_endpoint
python_connect_endpoint
send_hwm
receive_hwm
connect_timeout
ack_timeout
retry_interval
max_pending_results
max_retries_or_retention_policy
reconnect_backoff
runtime_instance_id
~~~

HWM 只约束 bounded resource，不构成 delivery guarantee。pending store 满时必须 fail-closed，并产生明确 INTERNAL_ERROR/RESULT_QUEUE_FULL 证据；不能丢 result 后继续报告成功。

### 7.3 disconnect/reconnect

1. 连接建立时交换 schema_version=4、runtime_instance_id、protocol hash 和 channel role。
2. handshake 不匹配立即 fail-fast；不能让 v4 Unity 和 v3 bridge 静默工作。
3. bridge 重启后的 pending 恢复策略必须明确：持久化恢复，或 runtime fail-closed 并报告未确认 execution；不得重新执行 action。
4. reconnect 后重复 result 必须走相同 dedup 路径。

### 7.4 P0-L3.5 bridge runtime wiring

P0-L3.5 将第 7.1 节的 bridge broker 接入真实
`unity_bridge_node` 进程。当前 resolved topology 为：

~~~text
Unity DEALER  --connect-->  bridge ROUTER(bind result_ingress)
Python DEALER --connect-->  bridge ROUTER(bind result_relay)
~~

`state`、`depth` 和 command 的既有 PUB/SUB/command socket 不变。bridge
新增的默认端口解析为：

~~~text
execution_result_port = explicit value, otherwise cmd_port + 2
python_result_port    = explicit value, otherwise cmd_port + 3
execution_result_bind_host = 0.0.0.0 / local runtime resolved host
python_result_bind_host    = 127.0.0.1 by default
python_result_retry_interval_s = 0.5 by default; one retry eligibility per interval
~~

端口可以通过 launch/private parameters 覆盖；默认派生规则保证现有
multi-worker `cmd_port` 分配不会让 result ingress 端口互相冲突。result
socket 使用现有 `recv_hwm`/`send_hwm` resolved values，不把 HWM 当作可靠性
保证，也不改变 state/depth telemetry HWM。

Unity ingress 和 Python relay 都使用 MessagePack map。result map 包含第 5
节的语义字段以及 `result_payload_hash: bin32`；hash 仍只覆盖第 3.1 节的
15-field canonical payload。bridge 发给 Unity 的 ACK 为
`PrimitiveExecutionResultAck`，包含 schema、message type、runtime identity、
execution identity、`DURABLE_RECEIVED`、result hash 和 command hash。

Python client 先发送：

~~~text
{schema_version: 4, message_type: "PrimitiveExecutionResultReady"}
~~~

bridge 收到 Ready 后，重试当前 memory pending store 中仍处于
`RESULT_SENT_PENDING_RECEIPT` 的 immutable result。收到 receipt 后停止 relay
重试，但保留 result 直到 Python commit。Python commit 使用：

~~~text
PrimitiveExecutionResultCommit(
  schema_version,
  message_type,
  commit_status = "COMMITTED",
  runtime_instance_id,
  execution_id,
  result_payload_hash,
  command_sequence_hash
)
~~~

bridge 逐字段验证 commit identity，并返回
`PrimitiveExecutionResultCommitAck`，状态为 `COMMITTED`、`DUPLICATE` 或
`PROTOCOL_ERROR`。错误 hash、command identity、key 或 schema 不会清除
pending result。

Python 在每次合法 result application receive（包括 retransmit/duplicate）后
先发送 `PrimitiveExecutionResultReceiptAck(RECEIVED)`。bridge 只有收到与
`(runtime_instance_id, execution_id, result_payload_hash,
command_sequence_hash)` 完全匹配的 receipt 后，才结束
`RESULT_SENT_PENDING_RECEIPT`；receipt 丢失时 bridge 继续重发 immutable
result。transition/replay 仍由后续 `PrimitiveExecutionResultCommit` 进行
exactly-once dedup。

本轮 `InMemoryResultStore` 只保证 bridge 进程生命周期内的 durable receipt；
crash/restart 后的跨进程 pending 恢复仍为 `DEFERRED`，不能伪装成已实现。

## 8. Python step_primitive commit rule

旧规则：收到 frame 0..24 才成功。

v4 新规则：

~~~text
收到 PrimitiveExecutionResult
  status == COMPLETE
  execution_id == expected
  command_sequence_hash == submitted hash
  requested_frame_count == 25
  applied_frame_count == 25
  last_applied_frame_index == 24
  endpoint_state_id == first_applied_state_id + 24
  endpoint observation exact-match 可取得
=> transition valid, exactly-once commit
~~~

frame telemetry 的缺失不能单独使上述 result 失败。

以下任一情况必须 fail-closed，不构造虚假 s_{t+1}：

- COMPLETE 的 endpoint state 不存在；
- endpoint state id/time 与 observation 不匹配；
- command hash 不匹配；
- execution_id 不匹配；
- schema version 不匹配；
- result invariant 不满足。

## 9. Endpoint observation binding

禁止：

~~~text
COMPLETE 到达
-> 取之后看到的第一条 state
-> 当作 s_{t+1}
~~~

v4 必须使用精确 binding：

~~~text
endpoint_state_id
endpoint_sim_time_ns
endpoint_observation_ref
~~~

endpoint_observation_ref 必须绑定 `schema_version`、runtime、episode、reset、state、depth 和 sim time。推荐设计是 bridge/Unity 提供按完整 ObservationRef 的可靠 observation retrieval；如果已有状态缓存不足，则 result channel 必须携带构造 Python s_{t+1} 所需的核心 endpoint snapshot。

如果 endpoint observation 无法取得，Python 必须产生明确的 ENDPOINT_OBSERVATION_UNAVAILABLE 失败证据，不能用模糊 state、zero fill 或旧 observation commit transition。

### 9.1 P0-M1.5 explicit observation identity

一个可提交的 endpoint observation 由同一条 indexed record 组成，必须同时
具有以下字段：

| 字段 | 语义 |
|---|---|
| `runtime_instance_id` | worker/runtime 隔离身份 |
| `state_id` | physics endpoint state identity |
| `depth_id` | 明确的 depth capture identity；当前 wire result 的 `capture_id` 为其字符串表示 |
| `physics_time_ns` | 产生 `state_id` 的 FixedUpdate 时间 |
| `capture_time_ns` | depth render/readback capture 时间；允许与 physics time 不同 |
| `episode_id` | episode 边界 |
| `reset_id` | reset 边界 |

Unity 在 endpoint physics frame 完成时，为下一次明确的 depth capture 排队
上述 binding；depth capture 保存这组 identity 和自己的 capture timestamp。
只有这次显式绑定的 depth readback 完成后，Unity 才生成 COMPLETE。result
中的 `endpoint_observation_ref.depth_id` 必须等于该 `depth_id`，其
`state_id` 与 `sim_time_ns` 分别等于 `state_id` 与 `physics_time_ns`；
`episode_id`、`reset_id` 和 `runtime_instance_id` 必须来自同一显式 endpoint
binding。

现有 state/depth telemetry 的既有字段和索引保持不变；identity 作为追加
字段发布。Python indexed store 只能按 `state_id + depth_id` 精确匹配，并
同时校验 physics/capture time、runtime、episode 和 reset。不得使用 latest、
nearest、arrival order 或下一帧推断。缺 depth、identity mismatch 或时间/边界
mismatch 都不得产生 transition commit。

## 10. v3 -> v4 同步范围

实现阶段必须同步检查以下 module/interface：

| 层 | 文件/接口 |
|---|---|
| Unity | XMProtocol.cs、XMSimulationManager.cs、XMConfig.cs/resolved runtime config |
| wire schema | xm_protocol.hpp、MessagePack schema、canonical hash 定义 |
| bridge | unity_bridge_main.cpp、src/bridge/*、reliable result broker、pending/dedup store |
| ROS | 如仍需要 ROS exposure，新增 PrimitiveExecutionResult.msg / PrimitiveExecutionResultAck.msg，并明确 ROS 是否只是 adapter |
| planning | primitive_execution_contract.py、unity_env.py、execution_transport_audit.py |
| evaluator/training | 所有依赖旧 exact-25 receipt 的成功、terminal、replay/transition accounting 路径 |
| tests | unit、integration、disconnect/reconnect、12-worker identity isolation |

schema 版本统一为 4。任何层收到 v3 message 或发送 v3 result 都必须显式 SCHEMA_MISMATCH 并停止当前 execution；禁止兼容性猜测和 silent fallback。

## 11. Required RED -> GREEN tests

实现前先固定测试 seam；以下测试不得删除现有 execution transport audit：

1. COMPLETE 第一次送达；
2. COMPLETE 丢失后 result retransmission；
3. ACK 丢失导致 duplicate COMPLETE；
4. duplicate 不产生 duplicate transition/reward/replay/episode step；
5. REJECTED 立即返回，applied_frame_count=0；
6. CANCELLED 立即返回并保留 applied prefix；
7. FAILED 立即返回并保留 applied prefix；
8. wrong execution_id reject；
9. wrong command_sequence_hash reject；
10. schema mismatch reject/fail-fast；
11. endpoint_state_id mismatch reject；
12. frame telemetry 丢 1 条但正确 COMPLETE 仍能完成 execution accounting；
13. Unity 实际只 apply 24 时，不能生成声称 25 的 COMPLETE；
14. disconnect/reconnect 后 pending COMPLETE 原样恢复发送；
15. result hash 改变的 duplicate execution 必须 protocol error；
16. 12 个 runtime identity 相同 execution_id 不得互相 dedup。

## 12. Runtime validation gates

### 12.1 single worker

至少 10,000 completed primitives，统计：

~~~text
physical primitive requested
Unity applied 25/25
COMPLETE generated
COMPLETE durable-received
COMPLETE committed
ACK generated / ACK received
result retransmission count
duplicate result count
duplicate transition count
REJECTED / CANCELLED / FAILED count
frame telemetry drop count
protocol_error count
endpoint observation binding failures
~~~

Gate：

~~~text
Unity true execution 25/25       = 100%
valid completed result accounted  = 100%
duplicate transition              = 0
wrong execution association       = 0
protocol_error                    = 0
~~~

telemetry PUB 偶发 drop 允许，但不得影响 execution transaction correctness。

### 12.2 twelve workers

single-worker GREEN 后再跑 12 workers、至少 10,000 completed primitives，并按 worker 输出相同统计。额外要求：

~~~text
runtime identity collision = 0
execution/result mismatch = 0
duplicate transition      = 0
protocol_error             = 0
~~~

P0 GREEN 前禁止启动 RL/P1。

## 13. Manifest requirements

重新 freeze source/binary 时，manifest 必须记录：

~~~text
planning source tree SHA
Unity source tree SHA
XMflight.x86_64 SHA
Assembly-CSharp.dll SHA
unity_bridge_node SHA
primitive_execution_schema_v4.md SHA
protocol schema SHA
BC checkpoint SHA
motion primitive SHA
resolved result ports
resolved HWM/timeout/retry/max-pending values
fixedDeltaTime
timeScale
worker count
runtime_instance_id per worker
~~~

Unity identity 必须包含 Assembly-CSharp.dll，不能只记录 XMflight.x86_64。

## 14. Current implementation boundary

本次 P0-J/P0-K/P0-L1/P0-L2a/P0-L2b 阶段已经完成：

- 真实 RED 的最小 regression fixture；
- diagnoser first-loss precedence 修复；
- 4 个 synthetic boundary tests 和真实 fixture test；
- 本 v4 设计文档。
- Python schema-v4 pure model、typed canonical identity 和 invariant；
- language-neutral COMPLETE/REJECTED/CANCELLED/FAILED golden vectors；
- Python/C++/C# byte-for-byte canonical codec contract tests。
- Unity 侧纯 ACK typed model/codec、immutable result lifecycle 与 deterministic
  retry seam；
- physical execution exactly-once、result generation exactly-once、result
  transmission at-least-once 的 C# contract harness。

本次明确未完成：

- Unity runtime 中接入 result pending/retransmission seam；
- bridge reliable broker；
- Python dedup/transition commit 改造；
- ROS message 改造；
- P0-M integration runtime；
- P0-N/P0-N1/P0-N2 validation。

因此当前总状态仍为：P0 RED。P0-L2a 只证明三语言纯 codec/hash 一致，未证明 runtime transaction correctness。唯一已捕获的 runtime first failing gate 仍是旧协议的 UNITY_PUB_TO_BRIDGE frame 24 loss；不能作为 P0 GREEN 或 RL/P1 启动依据。
