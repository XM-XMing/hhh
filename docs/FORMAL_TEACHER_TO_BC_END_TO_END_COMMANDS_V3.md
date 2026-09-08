# Formal Teacher-to-BC End-to-End Commandbook V3

This is a user handoff only. The commands below were not run at formal scale.
They stop after BC training. Do not run them until the disk preflight passes.

Rules:

- Every Python pipeline command starts after conda activate xm.
- Use devel/; do not run catkin_make install and do not source an install tree.
- Use the current Player and current devel Bridge shown below.
- Formal collection worker profile is 12 because only 12 workers were measured
  in the fresh optimized benchmark.
- Do not reuse a partial run with a different run ID, training ID, or launch
  nonce. Resume is an explicit same-identity operation only.

## A. Full clean end-to-end path

Keep the following in one terminal. It is intentionally explicit so the
resolved configuration is recorded in each artifact.

### Step 0: environment and frozen variables

~~~bash
set -euo pipefail
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
source /opt/ros/noetic/setup.bash
source /home/xm/XM/xm_ws/devel/setup.bash

cd /home/xm/XM/xm_ws/src/planning
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"

export WS=/home/xm/XM/xm_ws
export ROOT=/home/xm/XM/xm_ws/src/planning
export UNITY=/home/xm/XM/xm_ws/src/unity/XMflight.x86_64
export UNITY_DATA=/home/xm/XM/xm_ws/src/unity/XMflight_Data
export BRIDGE=/home/xm/XM/xm_ws/devel/lib/planning/unity_bridge_node
export CACHE="$ROOT/data/map_data/forest_voxels_10cm.npz"
export MPL_NPZ="$ROOT/data/motion_primitives/motion_primitives_105.npz"
export MPL_JSON="$ROOT/data/motion_primitives/motion_primitives_105.json"

export FORMAL_RUN=flight_reliable_exact_teacher_seed2026_20260901_v9
export BASE="$ROOT/data/teach/$FORMAL_RUN"
export CAND="$BASE/mission_candidates.csv"
export MISSIONS="$BASE/missions.csv"
export ROUTES="$BASE/mission_routes"
export ROLL="$BASE/rollouts"
export LABELS="$BASE/teacher_labels.npz"
export MASKS="$BASE/depth_action_masks.npz"
export AUDIT="$BASE/dataset_audit.json"
export BC="$BASE/bc_mmap"
export BC_TRAIN="$BASE/bc_training"
export FORMAL_MAX_SAMPLING_ATTEMPTS=8000000
export FORMAL_SAFE_COLLECTION_WORKERS=12

export PLANNING_MOTION_PRIMITIVES_NPZ="$MPL_NPZ"
export PLANNING_MOTION_PRIMITIVES_JSON="$MPL_JSON"
export PLANNING_COLLISION_BACKEND=cpp_cpu
export PLANNING_GLOBAL_ROUTE_BACKEND=cpp_native
export PLANNING_DEPTH_SAFETY_BACKEND=cpp

# Native libraries resolve automatically from the project-derived
# "$WS/devel/lib" location.  The following are optional explicit overrides
# for a special benchmark or alternate test binary; they are not required for
# the formal flow:
# export PLANNING_COLLISION_LIBRARY="$WS/devel/lib/libplanning_collision_checker.so"
# export PLANNING_VOXEL_MAP_LIBRARY="$WS/devel/lib/libplanning_voxel_map.so"
# export PLANNING_GLOBAL_ROUTE_LIBRARY="$WS/devel/lib/libplanning_global_route.so"
# export PLANNING_DEPTH_SAFETY_LIB="$WS/devel/lib/libplanning_depth_safety.so"

test "$(which python)" = "$CONDA_PREFIX/bin/python"
test -x "$UNITY"
test -x "$BRIDGE"
test ! -e "$BASE"

free_bytes=$(df -B1 -P "$ROOT" | awk 'NR == 2 {print $4}')
awk -v free_bytes="$free_bytes" 'BEGIN { exit !(free_bytes >= 225.039 * 1024 * 1024 * 1024) }'
mkdir -p "$BASE"
~~~

The disk check is deliberately fail-closed. The measured current free space
was below this threshold when this commandbook was written.

### Step 1: generate and verify MPL

~~~bash
source /home/xm/anaconda3/etc/profile.d/conda.sh && conda activate xm && python scripts/generate_motion_primitives.py --config config/motion_primitives.yaml
test "$(sha256sum "$MPL_NPZ" | awk '{print $1}')" = 22ad22fe66a88633de6effff11c03aa4ca4e8df362f8b410e17effbd91ff7309
test "$(sha256sum "$MPL_JSON" | awk '{print $1}')" = c1b795e4737a12034b0d59a1b457f559e7ad138527df2c90ed35db1326e42ea8
~~~

Success condition: MPL has 105 actions, 25 command frames, 26 reference
samples, and the three formal hashes.

### Step 2: build and verify the deterministic voxel cache

~~~bash
source /home/xm/anaconda3/etc/profile.d/conda.sh && conda activate xm && python scripts/build_voxel_cache.py --map-bin "$ROOT/data/map_data/forest_point_cloud.bin" --file-frame unity --cache "$CACHE" --data-offset-bytes -1 --voxel-size 0.10 --inflate-radius 0.35
test "$(sha256sum "$CACHE" | awk '{print $1}')" = a2374091ccc12a26635d0e36294df965fa576bc3efe78066ca7b6cb4a0dd1691
~~~

### Step 3: prepare 100,000 V2 Teacher-audited missions

max-sampling-attempts is a global preparation attempt cap. The resolver uses
max(200000, 4 * max_candidates) when the value is zero; the explicit formal
value below is therefore 8,000,000.

~~~bash
source /home/xm/anaconda3/etc/profile.d/conda.sh && conda activate xm && python scripts/prepare_teacher_missions.py --candidate-index "$CAND" --missions "$MISSIONS" --route-store-prefix "$ROUTES" --required-passing 100000 --max-candidates 2000000 --max-sampling-attempts "$FORMAL_MAX_SAMPLING_ATTEMPTS" --seed 2026 --max-steps 45 --collision-cache "$CACHE" --voxel-size 0.10 --inflate-radius 0.35 --goal-distance 40.0 --min-start-valid-actions 40 --min-goal-valid-actions 10 --map-margin-m 3.0 --check-step 2 --global-route-resolution-m 0.25 --global-route-lookahead-m 3.0 --global-route-tracking-margin-m 0.0 --max-route-stretch 1.15 --workers 12 --max-inflight-results 24 --checkpoint-interval 500 --collision-threads 1 --progress-interval-sec 10 --overwrite
~~~

### Step 4: validate the mission and immutable RouteStore

~~~bash
source /home/xm/anaconda3/etc/profile.d/conda.sh && conda activate xm && python scripts/validate_teacher_missions.py --candidate-index "$CAND" --missions "$MISSIONS" --route-store-prefix "$ROUTES" --expected-passing 100000 --max-steps 45 --expected-seed 2026
~~~

Success condition: MISSION_COUNT=100000, unique mission IDs, V2 task SHA,
max steps 45, and a valid RouteStore identity.

### Step 5: runtime identity gate

~~~bash
source /home/xm/anaconda3/etc/profile.d/conda.sh && conda activate xm && python scripts/validate_pre_collection_runtime.py --unity-bin "$UNITY" --unity-data "$UNITY_DATA" --bridge "$BRIDGE" --point-cloud "$ROOT/data/map_data/forest_point_cloud.bin" --voxel-cache "$CACHE" --mpl-npz "$MPL_NPZ" --mpl-json "$MPL_JSON" --max-steps 45 --expected-observation-contract reliable_exact_endpoint_snapshot --manifest "$ROOT/docs/pre_collection_runtime_final_v3.json"
test "$(sha256sum "$UNITY" | awk '{print $1}')" = 61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365
test "$(sha256sum "$BRIDGE" | awk '{print $1}')" = 667c266e96211aabb5a0cd25b8b35b303903d35cbe2d95be8f9d2bbe5e63d789
~~~

### Step 6: formal 60,000 reliable-exact collection

The collector has a disk preflight and runtime warning/stop guard. It must
fail closed if the preflight does not pass.

~~~bash
source /home/xm/anaconda3/etc/profile.d/conda.sh && conda activate xm && python scripts/collect_rollouts_parallel.py --mission-index "$MISSIONS" --out-dir "$ROLL" --workers "$FORMAL_SAFE_COLLECTION_WORKERS" --target-accepted 60000 --max-steps 45 --unity-bin "$UNITY" --bridge "$BRIDGE" --collision-cache "$CACHE" --collision-backend cpp_cpu --depth-safety-backend cpp_native --collision-threads 1 --voxel-size 0.10 --inflate-radius 0.35 --prefetch-inflate-radius 0.40 --expected-planar-distance 40.0 --reliable-exact-timeout 10.0 --stream-horizon 1 --candidate-top-k 12 --current-check-step 1 --lookahead-check-step 2 --max-stream-drift-m 0.35 --max-endpoint-error-m 0.60 --max-actual-path-length-m 46.0 --max-sensor-skew-ms 80.0 --worker-ready-timeout 45.0 --startup-timeout 60.0 --window-width 320 --window-height 240 --master-base 25100 --cmd-base 25200 --depth-base 25600 --port-stride 20 --run-id "$FORMAL_RUN" --training-run-id "$FORMAL_RUN" --runtime-launch-nonce formal-20260901-v9-r1 --global-route-resolution-m 0.25 --global-route-lookahead-m 3.0 --global-route-tracking-margin-m 0.0 --beam-depth 3 --beam-width 8 --beam-branching 4 --beam-discount 0.95 --disk-warning-free-gb 30 --disk-stop-free-gb 15 --disk-check-interval-sec 15 --disk-safety-margin-gb 10
~~~

The collection gate requires accepted rows, reliable rows, zero legacy rows,
zero telemetry lookups, zero snapshot misses, zero skew, zero frame failures,
zero collector errors, valid endpoint identity, and matching runtime identity.

### Step 7: collection validation

~~~bash
source /home/xm/anaconda3/etc/profile.d/conda.sh && conda activate xm && python scripts/validate_teacher_collection.py --rollout-dir "$ROLL" --missions "$MISSIONS" --route-store-prefix "$ROUTES" --min-accepted 60000 --expected-workers 12 --max-steps 45 --expected-observation-contract reliable_exact_endpoint_snapshot
~~~

### Step 8: relabel accepted episodes

~~~bash
source /home/xm/anaconda3/etc/profile.d/conda.sh && conda activate xm && python scripts/label_teacher_rollouts.py --index "$ROLL/rollout_index.csv" --out-labels "$LABELS" --route-store-prefix "$ROUTES" --collision-cache "$CACHE" --voxel-size 0.10 --inflate-radius 0.35 --num-workers 12 --worker-chunksize 1 --progress-interval 100 --collision-backend cpp_cpu --collision-threads 1 --beam-depth 3 --beam-width 8 --beam-branching 4 --beam-discount 0.95 --global-route-resolution-m 0.25 --global-route-lookahead-m 3.0 --global-route-tracking-margin-m 0.0 --direct-mmap
~~~

### Step 9: depth action masks

~~~bash
source /home/xm/anaconda3/etc/profile.d/conda.sh && conda activate xm && python scripts/generate_depth_action_masks.py --index "$ROLL/rollout_index.csv" --out-masks "$MASKS" --route-store-prefix "$ROUTES" --num-workers 12 --progress-interval 100 --direct-mmap --depth-min-m 0.30 --depth-max-m 3.00 --unknown-depth-threshold-m 2.95 --collision-radius-m 0.40 --depth-slack-m 0.08 --path-sample-stride 4 --patch-radius-px 2 --max-patch-radius-px 14
~~~

### Step 10: dataset provenance audit

~~~bash
source /home/xm/anaconda3/etc/profile.d/conda.sh && conda activate xm && python scripts/audit_teacher_dataset.py --index "$ROLL/rollout_index.csv" --route-store-prefix "$ROUTES" --labels "$LABELS" --depth-masks "$MASKS" --expected-planar-distance 40.0 --planar-distance-tolerance 0.001 --max-endpoint-error-m 0.60 --max-sensor-skew-ms 80.0 --out-json "$AUDIT"
~~~

### Step 11: BC mmap and validation

~~~bash
source /home/xm/anaconda3/etc/profile.d/conda.sh && conda activate xm && python scripts/build_bc_mmap_dataset.py --index "$ROLL/rollout_index.csv" --labels "$LABELS" --depth-action-masks "$MASKS" --observation-contract reliable_exact_endpoint_snapshot --out-dir "$BC" --overwrite
source /home/xm/anaconda3/etc/profile.d/conda.sh && conda activate xm && python scripts/validate_bc_dataset.py --index "$ROLL/rollout_index.csv" --labels "$LABELS" --depth-action-masks "$MASKS" --dataset-audit "$AUDIT" --bc-mmap "$BC" --expected-observation-contract reliable_exact_endpoint_snapshot --mode fast --sample-count 10000
source /home/xm/anaconda3/etc/profile.d/conda.sh && conda activate xm && python scripts/validate_bc_dataset.py --index "$ROLL/rollout_index.csv" --labels "$LABELS" --depth-action-masks "$MASKS" --dataset-audit "$AUDIT" --bc-mmap "$BC" --expected-observation-contract reliable_exact_endpoint_snapshot --mode full --sample-count 10000
~~~

### Step 12: BC training stop point

The measured throughput-safe bounded profile is batch 256, 8 DataLoader
workers, and prefetch 4. The formal algorithm owner remains the existing BC
loss/config contract; batch size is explicit here as a measured runtime
profile. Use batch 128 instead when exact historical training comparability is
required.

~~~bash
source /home/xm/anaconda3/etc/profile.d/conda.sh && conda activate xm && python scripts/train_soft_bc.py --index "$ROLL/rollout_index.csv" --labels "$LABELS" --depth-action-masks "$MASKS" --dataset-cache "$BC" --out-dir "$BC_TRAIN" --epochs 40 --batch-size 256 --lr 3e-4 --weight-decay 1e-4 --val-ratio 0.20 --seed 2026 --device cuda --num-workers 8 --prefetch-factor 4 --cpu-threads 24 --disable-tf32 --depth-history-frames 1 --ce-weight 1.0 --kl-weight 0.3 --ce-target teacher --loss-mask height_depth --deployment-safety-mask depth --deployment-execution-mode continuous --grad-clip 5.0 --save-plots --disable-tensorboard --observation-contract reliable_exact_endpoint_snapshot --expected-planar-distance 40.0 --planar-distance-tolerance 0.001
~~~

Stop here. Policy evaluation is the next stage; AWAC/SAC is not part of this
commandbook.

## B. Current-fast formal path using the validated mission/RouteStore

When the existing validated 100K mission and RouteStore are intact, skip Steps
1–4 and set these variables after Step 0. The new run must still use a fresh
output directory, a fresh FORMAL_RUN, and a fresh launch nonce.

~~~bash
export SRC_RUN=flight_reliable_exact_teacher_seed2026_20260829_v4
export SRC_BASE="$ROOT/data/teach/$SRC_RUN"
export MISSIONS="$SRC_BASE/missions.csv"
export ROUTES="$SRC_BASE/mission_routes"
export FORMAL_RUN=flight_reliable_exact_teacher_seed2026_20260901_v9
export BASE="$ROOT/data/teach/$FORMAL_RUN"
export ROLL="$BASE/rollouts"
export LABELS="$BASE/teacher_labels.npz"
export MASKS="$BASE/depth_action_masks.npz"
export AUDIT="$BASE/dataset_audit.json"
export BC="$BASE/bc_mmap"
export BC_TRAIN="$BASE/bc_training"

source /home/xm/anaconda3/etc/profile.d/conda.sh && conda activate xm && python scripts/validate_teacher_missions.py --candidate-index "$SRC_BASE/mission_candidates.csv" --missions "$MISSIONS" --route-store-prefix "$ROUTES" --expected-passing 100000 --max-steps 45 --expected-seed 2026
~~~

Then execute Steps 5–12 with the new BASE and the same identity gate. Do not
point the new collector at an existing rollout directory.

## C. Second-terminal monitor

Run this in a second terminal after exporting the same run variables:

~~~bash
set -euo pipefail
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
source /opt/ros/noetic/setup.bash
source /home/xm/XM/xm_ws/devel/setup.bash
cd /home/xm/XM/xm_ws/src/planning
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export ROOT=/home/xm/XM/xm_ws/src/planning
export FORMAL_RUN=flight_reliable_exact_teacher_seed2026_20260901_v9
export ROLL="$ROOT/data/teach/$FORMAL_RUN/rollouts"

source /home/xm/anaconda3/etc/profile.d/conda.sh && conda activate xm && python scripts/monitor_teacher_collection.py --rollout-dir "$ROLL" --interval-sec 30
~~~

The monitor must show accepted progress, reliable rows, all quality counters,
12 worker states, disk usage, and process state. Stop only if the collector's
fail-closed thresholds or system responsiveness gate requires it.
