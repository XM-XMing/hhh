# C++ Bridge Refactor Phase 3

This phase changes Bridge ownership only. Python, training logic, protocol schemas, ROS topics, ports, ZMQ endpoints, retry intervals, telemetry semantics, collision backends, and the C++ Global A* implementation are unchanged from Phase 2.

## Ownership

- `UnityBridgeNode`: configuration, module composition, poll order, aggregate metrics output.
- `CommandGateway`: ROS command subscriptions, legacy command publishing, reliable primitive command broker, command audit and retry state.
- `ResetGateway`: reliable reset request/complete/ACK state and pending reset store.
- `ResultGateway`: execution result store/broker, Python result peer state, relay retry state and result diagnostics.
- `SnapshotGateway`: endpoint snapshot cache, retrieval broker, Python snapshot requests and pending deliveries.
- `TelemetryBridge`: state/depth subscribers, ROS state/odom/depth publication, TF publication, telemetry audit and execution transport audit.
- `BridgeTransport`: ZMQ context and sockets only.

## Static validation

- C++17 syntax-only compilation with interface stubs: 7/7 Bridge translation units passed.
- ROS private parameter names: 36/36 preserved in the same order.
- Spin polling stages: 11/11 preserved in the same order.
- Result metrics JSON keys: 27/27 preserved.
- `ResetGateway` moved methods: 7/7 method bodies match Phase 2 after owner/config renaming.
- `TelemetryBridge` moved methods: 6/6 method bodies match Phase 2 after owner/config renaming.
- `CommandGateway`: 8/10 existing methods are body-identical after owner renaming; the remaining two only delegate reset ownership to `ResetGateway`.
- `ResultGateway`: changes are limited to snapshot delegation and aggregate-metrics callback.
- `SnapshotGateway`: changes are limited to state ownership and aggregate-metrics callback.
- Gateway source files containing `UnityBridgeNode::` methods: 0.
- `UnityBridgeNode` owns ROS publishers/subscribers, broker stores, pending reset/snapshot maps, or result/command retry state: 0.
- Python files changed relative to Phase 2: 0.
