# C++ ABI and artifact manifest

Frozen P manifest for C7 on 2026-08-28.  Paths below are relative to
`/home/xm/XM/src` unless stated otherwise.  O and Unity were not modified.

## Required public C symbols

The required symbol gate contains exactly 20 exported `planning_` symbols:

```text
planning_voxel_map_create
planning_voxel_map_destroy
planning_collision_create
planning_collision_create_from_voxel_map
planning_collision_destroy
planning_collision_radius
planning_collision_set_threads
planning_collision_check_path
planning_collision_check_action
planning_collision_check_actions_batch
planning_collision_check_pose_actions_batch
planning_global_route_create
planning_global_route_create_from_voxel_map
planning_global_route_destroy
planning_global_route_shape
planning_global_route_copy_blocked
planning_global_route_plan
planning_global_route_last_point_count
planning_global_route_copy_last_route
planning_depth_safety_mask
```

```text
REQUIRED_SYMBOL_COUNT=20
MISSING_REQUIRED_SYMBOL_COUNT=0
UNEXPECTED_LEGACY_SYMBOL_COUNT=0
ABI_SYMBOL_GATE=PASS
```

No formal exports matching the old CUDA, hybrid, Python fallback, or flat
Bridge implementation names were found.

## Devel-space library hashes

These are the canonical hashes from the clean Release devel build
`/tmp/xm-c7-clean-1.vLxvbk`:

| Artifact | SHA-256 |
| --- | --- |
| `libplanning_voxel_map.so` | `745ce457ccbca218e06cf3650ee1f7abffe42036ef90eb57c03a0497a8aaba7f` |
| `libplanning_collision_checker.so` | `fa9f82c6916bf4750994dbb707c80d35926804726bc63d293a3b611b8620cb94` |
| `libplanning_global_route.so` | `8547cda1c049964861f25e531a94017508993a7e06844087a1bd7c823cb5d5e0` |
| `libplanning_depth_safety.so` | `179e165cc2ef0f47232b61c0ccdb4ad1561047aab29893b841f4f8566cf27fab` |
| `unity_bridge_node` | `63b00f6a0b31aa72ace3e1c437477e7ee8b670934386f3bb1d645745c302d9ed` |

Install-space hashes are recorded separately because install strips the
temporary RUNPATH:

```text
INSTALL_BRIDGE_SHA256=4c18dabd900ce0ef0b098368e01dcd702aac8f239beb6b4e3bc636e9347979e9
INSTALL_VOXEL_SHA256=745ce457ccbca218e06cf3650ee1f7abffe42036ef90eb57c03a0497a8aaba7f
INSTALL_COLLISION_SHA256=49bc19779632fc14e2ee046d9d07facb6a15df43b16892a80acf0955cb586c6b
INSTALL_GLOBAL_ROUTE_SHA256=04cb194734efa3cef6c98e39ac3d6c51dc2018ffcf4d85f0bace0a7d0155917e
INSTALL_DEPTH_SAFETY_SHA256=179e165cc2ef0f47232b61c0ccdb4ad1561047aab29893b841f4f8566cf27fab
```

## Normalized public ABI hashes

Each hash is SHA-256 of the sorted `nm -D --defined-only` exported names
beginning with `planning_`, joined with newline separators:

```text
VOXEL_PUBLIC_ABI_SHA256=532341fa45a4aef654b34cb83ac7a72345f3d12c626bacd7c651cf4824f7f921
COLLISION_PUBLIC_ABI_SHA256=15fec8408de20d3ed9a1908651343e99ad08d252f6606d5b154d1980320fa381
GLOBAL_ROUTE_PUBLIC_ABI_SHA256=48d577020640942550df7bf223e7e64c1bdb0e1ad8498fb39f71e63bdb175f27
DEPTH_SAFETY_PUBLIC_ABI_SHA256=d168b4d7f43a3bbd329096cafc20419190ea3e687095852ff5840b97f47bec34
```

## Source and contract manifests

The C++ source manifest covers `CMakeLists.txt`, `package.xml`, and 60
headers/sources under `include/` and `src/` (62 files total) using sorted
relative filenames plus file bytes:

```text
CXX_SOURCE_MANIFEST_FILE_COUNT=62
CXX_SOURCE_MANIFEST_SHA256=2e940edac81555bf0f125b5059a93db951c355ece075cc18e22cf37edae88a77
BRIDGE_SOURCE_MANIFEST_FILE_COUNT=45
BRIDGE_SOURCE_MANIFEST_SHA256=27fc44226fe033a0dd63bcfb1fbf1326022684f9b06e479d1af7ab08e40acf9e
```

Contract source hashes:

```text
PORT_CONTRACT_SOURCE=python/planning/runtime/ports.py
PORT_CONTRACT_SHA256=b0ff7a9a185301e7ad221f074b5ee2101192230e347252fb9fd7f9e84d8e66f8
ZMQ_OPTION_CONTRACT_HEADER_SHA256=5f7bb0a46b04fc817afe9871025767bb763042bd3b225ce3e5194b46184c1ec5
ZMQ_OPTION_CONTRACT_IMPLEMENTATION_SHA256=8d2f5445dc8287219a1281b89bba7b30baa7c9ce6ed70308a109bac4db7f7a9b
```

## Dependency and toolchain gate

Collision depends on `libplanning_voxel_map.so`, `libgomp.so.1`, and standard
libraries.  Global Route depends on the shared VoxelMap and standard
libraries.  Bridge depends on ROS Noetic, `libzmq.so.5`, OpenSSL crypto,
msgpackc, and standard libraries.  `ldd`/`readelf` found no CUDA or hybrid
collision dependency.  CUDA runtime validation remains
`DEFERRED / NOT_AVAILABLE`.
