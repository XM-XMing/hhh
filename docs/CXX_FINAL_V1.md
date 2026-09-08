# C++ Final V1 Contract

Status: `CXX_FINAL_V1=PASS` on 2026-08-28.

This document is the compact operational contract for the frozen optimized
Planning C++ foundation at `/home/xm/XM/src`.  The original tree at
`/home/xm/XM/xm_ws/src/planning` remained the read-only behavior baseline.

## Formal owners

| Module | Formal P owner | Backend | Ownership rule |
| --- | --- | --- | --- |
| Voxel map | `planning::VoxelMap` | immutable C++17 shared map | one loaded occupancy buffer; readers do not copy full occupancy |
| Collision | `planning::CollisionChecker` | C++17/OpenMP CPU | consumes shared VoxelMap; no automatic Python fallback |
| Global Route | `planning::GlobalRoutePlanner` | native C++17 | consumes shared VoxelMap; native route is the formal path |
| Depth Safety | native depth-safety library | C++17 | preserves 105-action mask and clearance contract |
| Bridge | `planning::bridge::UnityBridgeNode` | split C++17/ROS/ZMQ | one formal implementation behind `unity_bridge_node` |

CUDA/hybrid collision is outside the formal P capability.  Its runtime was
not available for validation; the status is `DEFERRED / NOT_AVAILABLE`, and
the scope decision is `OUT_OF_FORMAL_SCOPE`, not `PROVEN_INCORRECT`.  This
does not affect PyTorch CUDA, AMP, or GradScaler used by BC/AWAC.

## Public runtime rules

- Native library creation or execution failure is fail-closed.
- Python reference implementations are explicit test/debug tools only.
- Public C ABI names, output buffers, row order, float calculations, and
  action ordering are frozen.
- Collision, route, and depth consumers share the immutable VoxelMap.
- Bridge command/result/snapshot/reset/telemetry payload and receipt contracts
  remain the v4 contracts.
- A missing `data/motion_primitives` source directory must not make a clean
  configure or install fail; CMake skips that optional install tree.

## Build and runtime environments

The supported reproducible split is:

1. `conda activate xm` for Release configure/compile, devel-space checks, and
   native Python import.
2. `/usr/bin/python3` for the ROS/Catkin install-space build because the conda
   distutils helper rejects Catkin's `--install-layout=deb` option.

Both clean builds compiled and linked the complete target set.  The install
space was imported from conda after installation and loaded the installed
native libraries successfully.  This toolchain split is intentional evidence,
not a source-level fallback.

## Freeze evidence

```text
C7_FINAL_CXX_FREEZE=PASS
CXX_FROZEN_FOR_PLANNING=YES
P_FULL_BUILD=PASS
PYTHON_NATIVE_IMPORT=PASS
ABI_SYMBOL_GATE=PASS
COLLISION_REGRESSION=PASS
GLOBAL_ROUTE_REGRESSION=PASS
DEPTH_SAFETY_REGRESSION=PASS
PROTOCOL_BRIDGE_REGRESSION=PASS
ONE_WORKER_RUNTIME_SMOKE=PASS
TWELVE_WORKER_RUNTIME_SMOKE=PASS
SINGLE_WORKER_FAULT_ISOLATION=PASS
PROCESS_CLEANUP=PASS
ALGORITHM_CHANGED=NO
```

Canonical frozen Unity artifacts:

```text
PLAYER_SHA256=61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365
PLAYER_RUNTIME_ASSEMBLY_SHA256=7ffb7bab78d84b45980b17daabae8872b8ba772af2821aa9f8a7da4c54a6dabc
```

The editor-only assembly hash `11f843ba787566ba1838d47d205978646b6d5eee0126f63260a6352a82551986`
must not be used as the Player runtime identity.

## Boundaries after freeze

This freeze does not authorize Python pipeline refactoring, reliable Teacher
multi-worker collection, BC checkpoint/training changes, evaluation changes,
AWAC/RL changes, Unity changes, formal data collection, or commits.  Those
remain the next controlled workstream.
