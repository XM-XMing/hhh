# Formal Teacher-to-BC End-to-End Commandbook V3.1

This is a gated decision record for the W20 Collection-worker promotion. The
V3 commandbook remains the executable commandbook and is intentionally not
rewritten because the V3.1 promotion gate failed.

~~~text
FORMAL_COLLECTION_W20_PROMOTION_V3_1 = FAIL
FORMAL_SAFE_COLLECTION_WORKERS = 12
OTHER_PIPELINE_WORKER_CONFIG_CHANGED = NO
FORMAL_60K_EXECUTED = NO
~~~

## Decision

The bounded W20 run reached 420 accepted episodes from 469 attempts with 20/20
workers ready and finished. Reliable-exact quality, endpoint identity,
physical port uniqueness, cross-talk, orphan cleanup, disk watchdog, and
collection validation passed. The hard resource gate did not pass: swap usage
was observed at 0.697266 GiB, and the locale-sensitive monitor did not retain a
complete available-memory/swap time series. Therefore no formal worker
promotion is authorized.

## Canonical commandbook status

Use the V3 commandbook for any later user-authorized operation:

`docs/FORMAL_TEACHER_TO_BC_END_TO_END_COMMANDS_V3.md`

Its formal Collection setting remains:

~~~text
export FORMAL_SAFE_COLLECTION_WORKERS=12
python scripts/collect_rollouts_parallel.py ... --workers "$FORMAL_SAFE_COLLECTION_WORKERS" ...
~~~

All Python commands in that commandbook begin after `conda activate xm`, use
the canonical `devel/` tree, and do not source or build `install/`. Mission,
label, mask, audit, BC mmap, and BC DataLoader worker/config settings remain
their independent V3 settings.

## W20 evidence

~~~text
SMOKE_OUTPUT=/tmp/xm-formal-w20-promotion-v31
W20_ACCEPTED=420
W20_ATTEMPTED=469
W20_ACCEPTED_PER_SEC_WALL=0.798889164
W20_ACCEPTANCE_RATE=0.895522388
W20_READY=20/20
W20_FINISHED=20/20
W20_QUALITY_COUNTERS=ALL_ZERO
W20_PORT_COLLISIONS=0
W20_CROSS_TALK=0
W20_ORPHAN_PROCESSES=0
W20_GRACEFUL_CLEANUP=PASS
W20_DISK_WATCHDOG=PASS
W20_RAM_HEADROOM_SPOT_GIB=36.399048
W20_SWAP_USED_OBSERVED_GIB=0.697266
W20_PROMOTION=NOT_GRANTED
~~~

No formal 60K collection, formal 100K mission audit, BC training, or AWAC/SAC
run was executed as part of this audit.

## Re-opening the promotion gate

A later audit must first establish zero swap and a locale-independent complete
resource monitor before changing the Collection worker value. Until a new
audit passes all hard gates, the only formal Collection worker value is 12.

