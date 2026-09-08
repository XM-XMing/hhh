# Formal Teacher-to-BC End-to-End Commandbook V3.2

This commandbook is the current stage-specific profile after bounded offline
worker benchmark V2. The complete environment, identity gates, and unchanged
stage commands remain in `docs/FORMAL_TEACHER_TO_BC_END_TO_END_COMMANDS_V3.md`.
The historical V3.1 Collection decision record is preserved as fail-closed
evidence; the current V3.2 Collection profile supersedes its worker gate.

## Worker profile

~~~text
MISSION_PREPARATION_WORKERS=20
MISSION_PREPARATION_MAX_INFLIGHT_RESULTS=40
MISSION_COLLISION_THREADS=1
LABEL_NUM_WORKERS=20 (canonical config/default)
MASK_NUM_WORKERS=20 (canonical config/default)
FORMAL_SAFE_COLLECTION_WORKERS=20
BC_DATALOADER_WORKERS=8
BC_BATCH_SIZE=256
BC_PREFETCH_FACTOR=4
~~~

Mission Preparation, Label, depth-mask, and Collection worker decisions are
stage-specific. The bounded V2 benchmark promoted Label and depth masks to 20
workers. The user has now accepted Collection=20 based on the historical
near-20-hour W20 long-run success evidence. The earlier V3.1 W20 promotion
attempt remains recorded as fail-closed after observing 0.697266 GiB swap; it
is not erased or reclassified. BC DataLoader settings remain at their existing
V3 values. Do not use one stage's worker value to change another stage.

~~~text
FORMAL_COLLECTION_W20_PROMOTION_V3_1=HISTORICAL_FAIL_CLOSED
HISTORICAL_W20_SWAP_USED_GIB=0.697266
COLLECTION_W20_USER_ACCEPTANCE=YES
COLLECTION_W20_ACCEPTANCE_BASIS=HISTORICAL_NEAR_20_HOUR_LONG_RUN_SUCCESS
~~~

The fixed V2 fixture was
`/tmp/xm-label-mask-worker-benchmark-v2/fixture500`: 500 accepted episodes,
14,290 transitions, and 500 hard-linked NPZ files with zero duplicated source
bytes. Its source index was validated with the reliable exact endpoint
observation contract; the source collection artifacts were not modified.

## Mission command delta

After the same environment and identity gates from V3, the current formal
Mission Preparation command uses the following worker arguments:

~~~bash
python scripts/prepare_teacher_missions.py \
  --candidate-index "$CAND" \
  --missions "$MISSIONS" \
  --route-store-prefix "$ROUTES" \
  --required-passing 100000 \
  --max-candidates 2000000 \
  --max-sampling-attempts "$FORMAL_MAX_SAMPLING_ATTEMPTS" \
  --seed 2026 \
  --max-steps 45 \
  --collision-cache "$CACHE" \
  --workers 20 \
  --max-inflight-results 40 \
  --checkpoint-interval 500 \
  --collision-threads 1 \
  --overwrite
~~~

The remaining Mission geometry, route, Teacher, and progress arguments are
the V3 commandbook values and must remain explicit there. The command is a
handoff only; this benchmark did not execute the formal 100K Mission run.

## Stage-specific downstream commands

Keep the stage-specific worker settings explicit when running a formal stage:

~~~text
Collection: --workers "$FORMAL_SAFE_COLLECTION_WORKERS"  # 20
Collection validation: --expected-workers 20
Label: --num-workers 20
Depth masks: --num-workers 20
BC: --num-workers 8 --batch-size 256 --prefetch-factor 4
~~~

All Python commands must run after:

~~~bash
source /home/xm/anaconda3/etc/profile.d/conda.sh
conda activate xm
source /opt/ros/noetic/setup.bash
source /home/xm/XM/xm_ws/devel/setup.bash
cd /home/xm/XM/xm_ws/src/planning
~~~

The devel setup owns the active `planning` import path. Do not add a manual
`PYTHONPATH=python` prefix or source the install space.

Do not run formal collection, full Label/Mask processing, BC training, or
AWAC/SAC as part of this benchmark decision.
