# O preservation manifest

Date: 2026-08-28

`O=/home/xm/XM/xm_ws/src/planning` is the read-only historical behavior and
artifact baseline. This manifest was produced for PY5 and does not authorize
any write, deletion, overwrite, or cutover in O.

## Gate

```text
O_PRESERVATION_MANIFEST=PASS
O_DATA_DELETION_COUNT=0
O_HISTORY_DELETION_COUNT=0
O_GIT_STATUS=NOT_AVAILABLE_NO_GIT_METADATA_AT_O
```

Directory `MANIFEST_SHA256` values are hashes of the deterministic sorted
`relative-path<TAB>file-size` listing. They are inventory hashes, not a
content hash of the 213 GB history archive. Individual map files have a
content SHA256 below. The inventory was read without modifying O.

## Required preservation inventory

| PATH | TYPE | SIZE (bytes) | FILE_COUNT | MANIFEST / SHA256 | CUTOVER_POLICY |
|---|---|---:|---:|---|---|
| `data/` | directory | 297852552 | 1 | `71e21ea091be6c075651d3bb727497c530c40081a30188af2cb45422450fef7b` | `PRESERVE` |
| `data/map_data/forest_point_cloud.bin` | file | 297819784 | 1 | `dfd87a5db1ab98276eda57e79de677f711da56f308a373b93e72484fa5876d40` | `PRESERVE` |
| `data_back_20260826/` | directory | 212992388097 | 120178 | `6b600703263bf3b7f0916fe61a92deaa869278fa99ebb16ccb654bd324e84f41` | `PRESERVE` |
| `data_back_20260826/awac/` | directory | 45259716584 | 20858 | `c8a6b334983abf34ce7dd593ee476bed05700cf7eae7803559ff8e7445af713e` | `ARCHIVE` |
| `data_back_20260826/bc/` | directory | 73250821649 | 42 | `034d6abe0ddeb03e1913eb4aec765785fb3a503617a031234c9b8cfbf01ce0e9` | `ARCHIVE` |
| `data_back_20260826/debug/` | directory | 922625082 | 1671 | `3f83c98b4bf346a48bb6556e2ff696cda0e15e94af78ae245da377d15970f75e` | `ARCHIVE` |
| `data_back_20260826/map_data/` | directory | 300610080 | 2 | `37e26cc27cd174f94ac4f4fc53cb75c0fffb8ea745188befc0dd83a568975b18` | `PRESERVE` |
| `data_back_20260826/map_data/forest_point_cloud.bin` | file | 297819784 | 1 | `dfd87a5db1ab98276eda57e79de677f711da56f308a373b93e72484fa5876d40` | `PRESERVE` |
| `data_back_20260826/map_data/forest_voxels_10cm.npz` | file | 2786200 | 1 | `a2374091ccc12a26635d0e36294df965fa576bc3efe78066ca7b6cb4a0dd1691` | `PRESERVE` |
| `data_back_20260826/motion_primitives/` | directory | 2454451 | 3 | `a339425142a45d69cecb9482672795b55d1caae7ff26a1162d13f3d7e0f5c2f8` | `ARCHIVE` |
| `data_back_20260826/sac/` | directory | 17476694664 | 6081 | `5ead68dd98155577b9385218d521aae8e483e5ee1e3fd1c745ff091d925836c7` | `ARCHIVE` |
| `data_back_20260826/smoke/` | directory | 67555512 | 868 | `eb6c3916e71e62d6895b20cbe70420c60f56b83a5a977d3608125b4221836cff` | `ARCHIVE` |
| `data_back_20260826/teach/` | directory | 75677396793 | 90515 | `2fa181a744e80051c37a9720ae0765ed08b9e1c57d5c0688dd91d4a58d639693` | `ARCHIVE` |
| `data_back_20260826/test/` | directory | 34509186 | 138 | `a51b25f5119ae16c5873cb1d33987620a7a21e6bd53c855a5561660781566c2a` | `ARCHIVE` |
| `docs/` | directory | 79200 | 5 | `aec7a62b1d5890ec59b49a74379fc443d1cfd8cfe8dc4bf46326c5b816daf636` | `ARCHIVE` |
| `include/` | directory | 141191 | 8 | `67cdbf54aee8e8553bd22f2cf1b06942b5762d863761e6f21f11fb2385b4f94f` | `ARCHIVE` |
| `launch/` | directory | 15959 | 8 | `06aab33332ea893b3b48e5efba3436218a15ae967d86c3a35dec76d01d82e532` | `ARCHIVE` |
| `msg/` | directory | 4683 | 3 | `09f1a95f609164862c0a1f6890996d06642b6dd579b3157a494677abbf920e54` | `ARCHIVE` |
| `config/` | directory | 6167 | 2 | `3cb7ec9f85b69a3d64cbc3301b6d529da5dd84f2074e30150f06ebd333e2869d` | `ARCHIVE` |
| `python/` | directory | 3024858 | 210 | `f031128352b7788eb870cf48b0a3a757815c35ba64eadfc13d70c3a3dbb641fb` | `ARCHIVE` |
| `scripts/` | directory | 193464 | 68 | `8bcbe93e1b5f12ef86093f5b4925381bbd08814bd9d22cffd33d978196230a09` | `ARCHIVE` |
| `src/` | directory | 196973 | 6 | `133536469904ed0575a09420f294ae2c316f03a02692360cd42ea0892f2565a7` | `ARCHIVE` |
| `tests/` | directory | 2450020 | 312 | `7b5c557b41f9666603d2a557b4631ea925367ae82c1f03453fa8c3724d38b2d` | `ARCHIVE` |
| `rviz/` | directory | 13692 | 3 | `6829fae3a08e8bd32820f48598fbb4ca701f28f9dba62712fbbccc433b7a8759` | `ARCHIVE` |
| `.benchmarks/` | directory | 4096 | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `EXCLUDE_GENERATED` |
| `.pytest_cache/` | directory | 90817 | 6 | `d843950c62b1ae12ceb97174ea04c3da42708f4de060b6247d382ab668a37912` | `EXCLUDE_GENERATED` |
| `build/`, `devel/`, `install/` | absent in O inventory | 0 | 0 | `ABSENT` | `REBUILD` |

The `awac`, `bc`, and `sac` history subtrees contain historical checkpoints,
evaluation and training evidence. `teach`, `smoke`, `test`, and `debug`
contain rollout, smoke, test, and diagnostic artifacts. No subtree was
selected for deletion based on its name.

## Untracked and user-owned artifacts

Neither O nor its parent was a Git worktree at audit time, so Git cannot
provide a reliable untracked-file list. The conservative policy is therefore
to preserve every O `data/` and `data_back_20260826/` file, the selected map
assets, and all existing historical artifact subtrees. The generated O cache
directories are excluded from the P sync rule; they are not deleted by PY5.

The P staged rehearsal used a copy of the O voxel asset only at
`/tmp/xm-planning-cutover-stage/data/map_data/forest_voxels_10cm.npz`. It did
not copy the O history archive and did not modify the source asset.

## Cutover invariant

The cutover dry-run excludes `/data/` and `/data_back_20260826/` from the P
source. The future formal MPL is generated in O after cutover from the copied
P configuration; it is not copied from the staged rehearsal or from history.
Any command that reports a deletion under either O data path or history path
must be rejected and the cutover gate remains `NO`.
