# RL Phase 1 Readiness and Calibration Certification V1

Status: `FAIL_BLOCKED_ENVIRONMENT`

This report records the Phase 1 implementation and bounded evidence without
opening formal Actor updates. The source tree is
`/home/xm/XM/xm_ws/src/planning`. Unity, Bridge, Final300, formal collection,
and standard AWAC long training were not started.

## Gate result

`PHASE1_READINESS=FAIL` because the current Codex runtime cannot create a
loopback socket (`PermissionError: [Errno 1] Operation not permitted`). The
independent host certification supplied for this checkout remains
`705 passed, 42 skipped, 0 failed`; it is not overwritten by the sandbox
result.

The required BC Dev100 baseline therefore remains absent. Its immutable target
is `data/test/awac_dev_seed4026_100`, whose manifest is a `DEV` artifact with
100 missions. The `data/test/bc_closed_loop_eval_300` artifact remains
`FINAL_TEST_ONLY` and was not executed or modified.

## Implemented contracts

- Actor depth learning rate changed from the blocked default `0.0` to the
  bounded-sanity-selected formal value `1.0e-6`.
- Critic depth learning rate remains `1.0e-5` and both Actor/Critic depth
  optimizer groups are required to be real, trainable groups.
- Candidate sanity values were `1.0e-6`, `2.5e-6`, and `5.0e-6`; all were
  finite and updated depth parameters. The smallest stable nonzero value was
  selected. This was an offline fixed-batch benchmark, not formal Actor
  training.
- `Phase1CalibrationSafetyCap` provides the typed maximum of 30,000
  transitions or 1,000 episodes. A `PENDING` gate at either bound becomes
  `BLOCKED_PENDING` and stops; the cap is not a required training amount.
- `Phase1CalibrationController` preserves the existing
  `PENDING`/`PASS`/`FAIL_DIVERGED` gate and never enables the Actor. The only
  Actor-enabling transition remains the explicit `PASS` transition in
  `Phase1StateMachine`.
- The state machine provides the calibration-to-standard transition, 10K/25K/
  50K `AWAC_ONLINE` checkpoint and DEV evaluation hooks, and the
  `BC Dev - 0.10` catastrophic-collapse stop without rollback.
- Calibration smoke remains critic-only: 160 transitions, 20 episodes, 8
  Critic updates, Actor update count 0, and gate `PENDING`. It is not formal
  calibration evidence and did not create `checkpoint_calibration_pass.pt`.

## Frozen identity and audit

| Item | Value |
|---|---|
| BC checkpoint | `data/teach/2026_6w/bc_training/checkpoint_best_soft.pt` |
| BC checkpoint SHA256 | `ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2` |
| Observation contract | `reliable_exact_endpoint_snapshot` |
| Task contract SHA256 | `2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df` |
| Reward contract SHA256 | `1eacab3788e9051e8c77d7c6aeafa2c50bd93dc09434b3adf4985850dd2b62d0` |
| Replay contract SHA256 | `2b6198b0d295a51687542e475505f85c0d84180a8e529b0c3ea4f84d99c2244f` |
| Phase-1 contract SHA256 | `6a3ae6e650db8e60b693396fbc3ac15bf6f6030c86dcd650dc1b6534a63c4233` |
| Action count / max steps | `105 / 45` |

The replay field set and source restrictions are unchanged. Static replay
audit evidence is clean: global-map leak count 0, Critic global-map leak count
0, and privileged replay field count 0.

## Verification

- `python -m compileall -q python scripts tests`: PASS.
- Phase-1 plus AWAC focused suite: `76 passed`.
- Sandbox full pytest: `706 passed, 42 skipped, 8 failed`; all 8 failures are
  loopback/socket permission failures in the pre-existing P0 runtime tests.
- Code-eligible full result after excluding those environment-only failures:
  `706 passed, 42 skipped, 0 failed`.
- Host full certification: `705 passed, 42 skipped, 0 failed` (user-supplied,
  retained as the host certification).
- `CODEX_SOCKET_ENVIRONMENT=RESTRICTED`; no socket workaround, test skip,
  mock, Unity, or Bridge launch was used.

## Changed files

- `python/planning/awac/phase1.py`
- `python/planning/awac/calibration.py`
- `python/planning/awac/checkpoint.py`
- `python/planning/awac/trainer.py`
- `scripts/benchmark_awac_actor_depth_lr.py`
- `tests/test_awac_phase1_readiness.py`
- `tests/contracts/test_contract_naming.py` (versioned contract inventory)

No BC, Unity, Bridge, Task, Observation, Reward, Motion Primitive, or Final300
artifact was changed. No commit was created.

## Remaining blockers and next action

1. On a socket-enabled host, run the one-time deterministic BC Dev100 closed
   loop and freeze its baseline manifest/SHA.
2. On the same host, run bounded formal critic-only calibration from the
   training missions. Reuse `evaluate_calibration_gate`; stop at PASS,
   `BLOCKED_PENDING` at 30,000 transitions/1,000 episodes, or
   `FAIL_DIVERGED`.
3. Only a genuine PASS may publish and resume-load
   `checkpoint_calibration_pass.pt`. Actor updates remain disabled in this
   report.

Next action: `RUN_ON_SOCKET_ENABLED_HOST_BC_DEV100_BASELINE_THEN_FORMAL_CRITIC_CALIBRATION`.
