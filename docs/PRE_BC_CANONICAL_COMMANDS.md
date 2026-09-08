# PY3 pre-BC canonical smoke commands

All commands in this record use the project environment requested for XMflight:

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
export PYTHONPATH=/home/xm/XM/src/python:/home/xm/XM/xm_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages
export PYTHONPYCACHEPREFIX=/tmp/xm-pre-bc-py3/pycache
export PYTHONDONTWRITEBYTECODE=1
```

The only generated output root was `/tmp/xm-pre-bc-py3/`. The O forest cache
was read-only input:

```bash
O_MAP=/home/xm/XM/xm_ws/src/planning/data_back_20260826/map_data/forest_voxels_10cm.npz
MPL_NPZ=/tmp/xm-pre-bc-py3/motion_primitives/motion_primitives_105.npz
MPL_JSON=/tmp/xm-pre-bc-py3/motion_primitives/motion_primitives_105.json
```

## Motion primitives, missions, and audit

```bash
python /home/xm/XM/src/scripts/generate_motion_primitives.py \
  --config /tmp/xm-pre-bc-py3/motion_primitives.yaml

python /home/xm/XM/src/scripts/generate_missions.py \
  --out-index /tmp/xm-pre-bc-py3/missions/candidates.csv \
 --num-episodes 200 --seed 55 --overwrite \
 --collision-cache "$O_MAP" --voxel-size 0.10 --inflate-radius 0.35 \
  --goal-distance 40.0 --route-candidate-factor 10.0 \
  --global-route-resolution-m 0.50 \
 --global-route-tracking-margin-m 0.10 --num-workers 4 --route-workers 4

python /home/xm/XM/src/scripts/audit_teacher_missions.py \
  --candidate-index /tmp/xm-pre-bc-py3/missions/candidates.csv \
  --out-index /tmp/xm-pre-bc-py3/missions/audited.csv \
  --required-passing 60 --max-steps 45 --stop-when-required --overwrite \
  --collision-cache "$O_MAP" --voxel-size 0.10 --inflate-radius 0.35 \
  --num-workers 4 --collision-threads 1 --collision-backend cpu \
  --beam-depth 3 --beam-width 8 --beam-branching 4 --beam-discount 0.95 \
  --global-route-resolution-m 0.50 --global-route-lookahead-m 12.0 \
  --global-route-tracking-margin-m 0.10
```

The generated MPL is new output. It is not copied from `data_back_20260826`.

## Reliable exact collection and merge

The passing collection used the direct P Python entrypoint and its positional
parallel supervisor interface. Managed values were supplied through the
environment because `collect_rollouts_parallel.py` intentionally exposes only
the mission-index, output-directory, and worker-count positional arguments:

```bash
export WORKSPACE=/home/xm/XM/xm_ws
export UNITY_BIN=/home/xm/XM/xm_ws/src/unity/XMflight.x86_64
export COLLISION_CACHE="$O_MAP"
export TARGET_ACCEPTED=20
export MAX_EPISODES=0
export MAX_STEPS=45
export COLLISION_THREADS=1
export EXPECTED_PLANAR_DISTANCE=40.0
export COLLECTION_RUN_ID=py3-pre-bc-smoke-p-20260828
export TRAINING_RUN_ID=py3-pre-bc-training-p-20260828
export RUNTIME_LAUNCH_NONCE=py3-launch-p-20260828
export RELIABLE_EXACT_TIMEOUT=10
export PLANNING_COLLISION_BACKEND=cpu
export PLANNING_VOXEL_MAP_LIBRARY=/tmp/xm-cxx-c2-2-p-devel/planning/lib/libplanning_voxel_map.so
export PLANNING_COLLISION_LIBRARY=/tmp/xm-cxx-c2-2-p-devel/planning/lib/libplanning_collision_checker.so
export PLANNING_GLOBAL_ROUTE_LIBRARY=/tmp/xm-cxx-c2-2-p-devel/planning/lib/libplanning_global_route.so
export COLLECTOR_EXTRA_ARGS="--beam-depth 3 --beam-width 8 --beam-branching 4 --beam-discount 0.95 --global-route-resolution-m 0.50 --global-route-lookahead-m 12.0 --global-route-tracking-margin-m 0.10"
export ROS_PACKAGE_PATH=/tmp/xm-pre-bc-py3/ros_overlay:/home/xm/XM/xm_ws/src:/opt/ros/noetic/share

python /home/xm/XM/src/scripts/collect_rollouts_parallel.py \
  /tmp/xm-pre-bc-py3/missions/audited.csv \
  /tmp/xm-pre-bc-py3/collection_p 2
```

Before this command, the P native libraries, P bridge binary, P-first ROS
overlay, runtime identity variables, and both MPL override variables were
exported. The shell wrapper was not used as the successful entrypoint because
its workspace bootstrap can resolve `planning` from O after sourcing the O
devel workspace. The collection itself performed the normal merge and emitted
`rollout_index.csv` plus `collection_summary.json`.

## Relabel, audit, depth mask, and BC mmap

```bash
python /home/xm/XM/src/scripts/label_teacher_rollouts.py \
  --index /tmp/xm-pre-bc-py3/collection_p/rollout_index.csv \
  --out-labels /tmp/xm-pre-bc-py3/labels/teacher_labels.npz \
  --collision-cache "$O_MAP" --voxel-size 0.10 --inflate-radius 0.35 \
  --num-workers 2 --worker-chunksize 1 --collision-backend cpu \
  --collision-threads 1 --compress \
  --beam-depth 3 --beam-width 8 --beam-branching 4 --beam-discount 0.95 \
  --global-route-resolution-m 0.50 --global-route-lookahead-m 12.0 \
  --global-route-tracking-margin-m 0.10

python /home/xm/XM/src/scripts/generate_depth_action_masks.py \
  --index /tmp/xm-pre-bc-py3/collection_p/rollout_index.csv \
  --out-masks /tmp/xm-pre-bc-py3/masks/depth_masks.npz \
  --num-workers 2 --compress --depth-min-m 0.30 --depth-max-m 3.00 \
  --unknown-depth-threshold-m 2.95 --collision-radius-m 0.40 \
  --depth-slack-m 0.08 --path-sample-stride 4 \
  --patch-radius-px 2 --max-patch-radius-px 14

python /home/xm/XM/src/scripts/audit_teacher_dataset.py \
  --index /tmp/xm-pre-bc-py3/collection_p/rollout_index.csv \
  --labels /tmp/xm-pre-bc-py3/labels/teacher_labels.npz \
  --depth-masks /tmp/xm-pre-bc-py3/masks/depth_masks.npz \
  --expected-planar-distance 40.0 --planar-distance-tolerance 0.001 \
  --max-endpoint-error-m 0.60 --max-sensor-skew-ms 80.0 \
  --out-json /tmp/xm-pre-bc-py3/dataset_audit.json

python /home/xm/XM/src/scripts/build_bc_mmap_dataset.py \
  --index /tmp/xm-pre-bc-py3/collection_p/rollout_index.csv \
  --labels /tmp/xm-pre-bc-py3/labels/teacher_labels.npz \
  --depth-action-masks /tmp/xm-pre-bc-py3/masks/depth_masks.npz \
  --observation-contract reliable_exact_endpoint_snapshot \
  --out-dir /tmp/xm-pre-bc-py3/bc_mmap --overwrite
```

The depth-mask multiprocessing workers require the same explicit MPL
override as the parent process:

```bash
export PLANNING_MOTION_PRIMITIVES_NPZ="$MPL_NPZ"
export PLANNING_MOTION_PRIMITIVES_JSON="$MPL_JSON"
```

## BC training smoke

```bash
python /home/xm/XM/src/scripts/train_soft_bc.py \
  --index /tmp/xm-pre-bc-py3/collection_p/rollout_index.csv \
  --labels /tmp/xm-pre-bc-py3/labels/teacher_labels.npz \
  --depth-action-masks /tmp/xm-pre-bc-py3/masks/depth_masks.npz \
  --dataset-cache /tmp/xm-pre-bc-py3/bc_mmap \
  --out-dir /tmp/xm-pre-bc-py3/bc_train --epochs 1 --max-episodes 21 \
  --batch-size 16 --seed 55 --device cpu --num-workers 0 --cpu-threads 2 \
  --disable-tf32 --deployment-safety-mask depth \
  --deployment-execution-mode continuous --no-amp --disable-tensorboard \
  --observation-contract reliable_exact_endpoint_snapshot \
  --expected-planar-distance 40.0 --planar-distance-tolerance 0.001
```

## Evaluation smoke

The managed wrapper's ordinary mode requires at least 100 episodes. To keep
PY3 bounded, the smoke invoked its explicit diagnostic mode once per selected
episode ID:

```bash
export EVAL_RELIABLE_V4=1
export EVAL_EXPECTED_OBSERVATION_CONTRACT=reliable_exact_endpoint_snapshot
export EVALUATOR_DEVICE=cpu
export EVAL_REPRO_MAX_STEPS=45
export EVAL_COLLISION_CACHE="$O_MAP"

EVAL_REPRO_EPISODE_ID=2 \
  /home/xm/XM/src/scripts/evaluate_policy_unity_managed.sh \
  /tmp/xm-pre-bc-py3/bc_train/checkpoint_best.pt \
  /tmp/xm-pre-bc-py3/missions/audited.csv \
  /tmp/xm-pre-bc-py3/evaluation/episode_000002 1
```

The same command was repeated for IDs `4,7,13,15,17,19,22,23,24`, with a
unique runtime identity and a fresh output directory each time. It used the
frozen Unity Player, the P native C++ bridge, and a P-first ROS overlay.

## Focused verification

```bash
python -m compileall -q /home/xm/XM/src/python/planning \
  /home/xm/XM/src/scripts /home/xm/XM/src/tests

python -m pytest -q -o cache_dir=/tmp/xm-pre-bc-py3/pytest-cache \
  --basetemp=/tmp/xm-pre-bc-py3/pytest-tmp \
  tests/test_observation_contract.py \
  tests/test_py0_reliable_exact_collection.py \
  tests/test_py0_collection_merge.py \
  tests/test_py2_relabel_mask_provenance.py \
  tests/test_bc_mmap_dataset.py \
  tests/test_bc_mmap_observation_provenance.py \
  tests/test_bc_training_observation_provenance.py \
  tests/test_evaluation_observation_provenance.py \
  tests/test_managed_policy_eval_contract.py \
  tests/test_depth_safety_production_cutover.py \
  tests/contracts/test_action_mask.py \
  tests/contracts/test_depth_safety.py

python /home/xm/XM/src/tests/compare_py2_relabel_mask_fixture.py \
  --o-labels /tmp/xm-py2-chain/o/teacher_labels.npz \
  --p-labels /tmp/xm-py2-chain/teacher_labels.npz \
  --o-masks /tmp/xm-py2-chain/o/depth_masks.npz \
  --p-masks /tmp/xm-py2-chain/depth_masks.npz
```

The focused suite result was `83 passed in 4.13s`; all ten canonical CLI help
commands also returned successfully.

## PY5 clean/staged validation (2026-08-28)

The PY5 canonical CLI gate ran all eleven entries from the staged P install
with `conda activate xm` and returned `11/11`. The formal MPL source config
was the P config itself; relative outputs were redirected by the P-first
staged ROS package root and remained under `/tmp/xm-planning-cutover-stage`.

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
export ROS_PACKAGE_PATH=/tmp/xm-planning-cutover-stage:/opt/ros/noetic/share
python3 /home/xm/XM/src/scripts/generate_motion_primitives.py \
  --config /home/xm/XM/src/config/motion_primitives.yaml
```

The command was run twice. It produced 105 actions, 25 command frames, 26
reference frames, contract SHA256
`f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d`, NPZ
SHA256 `22ad22fe66a88633de6effff11c03aa4ca4e8df362f8b410e17effbd91ff7309`,
and JSON SHA256
`c1b795e4737a12034b0d59a1b457f559e7ad138527df2c90ed35db1326e42ea8` on both
runs. The pair was not copied into P or O.

The staged PY5 chain is recorded at `/tmp/xm-py5-staged-e2e/` and passed the
bounded Pre-BC chain through a one-epoch BC smoke and audit-only reliable
evaluation. A separate 12-worker startup/identity/cleanup smoke passed with
12 unique runtime IDs, unique ports, reliable rows on every worker, zero
legacy rows, zero telemetry lookups, zero snapshot misses, zero skew, zero
frame failures, zero collector errors, and zero orphan processes. Its
`TARGET_ACCEPTED=0` and `MAX_STEPS=3` settings intentionally make it an
infrastructure smoke, not formal collection evidence.

The future P-to-O sync command, explicit excludes, backup, and rollback are
documented in `docs/P_TO_O_CUTOVER_PLAN.md`. It was only dry-run in PY5; no
formal collection or O write occurred.

## PY6 O canonical cutover validation

The PY6 cutover completed with O as the canonical import and runtime source.
Every command below is bounded validation evidence only; it does not start a
formal collection or formal training run.

```bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
WS=/home/xm/XM/xm_ws
O=$WS/src/planning
source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"
unset PYTHONPATH
export PYTHONPATH="$O/python:$WS/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages"
export ROS_PACKAGE_PATH="$O:$WS/src:/opt/ros/noetic/share"
export PLANNING_MOTION_PRIMITIVES_NPZ="$O/data/motion_primitives/motion_primitives_105.npz"
export PLANNING_MOTION_PRIMITIVES_JSON="$O/data/motion_primitives/motion_primitives_105.json"
```

The O Release/devel/install build passed with `/usr/bin/python3` for catkin
message generation. Python import resolved to `$O/python/planning`, and the
full suite returned `756 passed, 42 skipped, 0 failed`; canonical CLI help
returned `11/11`.

The bounded output root is `/tmp/xm-py6-o-e2e/`. The canonical sequence is:

```text
generate_motion_primitives.py
  -> generate_missions.py
  -> audit_teacher_missions.py
  -> collect_rollouts_parallel.py (2 workers, target 4 accepted, smoke only)
  -> label_teacher_rollouts.py
  -> audit_teacher_dataset.py
  -> generate_depth_action_masks.py
  -> build_bc_mmap_dataset.py
  -> train_soft_bc.py (one CPU epoch, smoke only)
  -> evaluate_policy_unity_managed.sh (one reliable audit-only episode)
```

The Unity/ROS smokes used the explicit frozen runtime identity:

```text
Player=61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365
Assembly-CSharp=7ffb7bab78d84b45980b17daabae8872b8ba772af2821aa9f8a7da4c54a6dabc
Bridge=63b00f6a0b31aa72ace3e1c437477e7ee8b670934386f3bb1d645745c302d9ed
Bridge=/tmp/xm-c7-clean-1.vLxvbk/devel/lib/planning/unity_bridge_node
```

The final 12-worker infrastructure command used an isolated port range,
`MAX_STEPS=3`, and `TARGET_ACCEPTED=0`; all workers completed with zero
provenance or collector failures. The complete cutover evidence and exact
hashes are in `docs/PY6_CUTOVER_REPORT.md`.
