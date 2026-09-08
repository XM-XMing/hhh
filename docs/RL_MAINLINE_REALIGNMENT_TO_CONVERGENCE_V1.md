# Mainline realignment: Proposed AWAC convergence development

Current phase is `PROPOSED_METHOD_CONVERGENCE_DEVELOPMENT`. The immediate
objective is a stable Dev100 result at or above 72, with collision, timeout,
dead-end, return, Actor health, Critic finiteness, replay audit, and runtime
provenance gates retained.

The frozen development comparison points are BC60K 69, default Standard AWAC
62 at `updates_per_step=0.50`, rejected Critic-LR03 61, and the best tuned
Standard control 65 at `updates_per_step=0.25`. The 65 result is not a
recovered Standard baseline.

The future mainline starts from Formal Critic Calibration V7, uses 16 workers,
seed 2026, 10K phase-local environment steps, `max_steps=45`, reliable-v4,
and `updates_per_step=0.25`. Existing frozen checkpoints and Replay are
comparison/source artifacts and are not modified.

The Foundation owners are:

- `planning.awac.confidence.TwinQConfidenceEstimator`
- `planning.awac.adaptive_bc_kl`
- `planning.awac.primitive_neighborhood`
- `planning.awac.primitive_exploration`

All innovation flags default to false, and the disabled path preserves the
existing Standard AWAC loss. The first future mainline run enables Confidence
and Adaptive BC KL only; primitive exploration follows only after that method
shows stable improvement. EXP1--EXP5 are templates/manifests, not runs.

`FINAL300_USED=NO`, `FORMAL_ABLATION_PHASE=DEFERRED`, and basic Standard AWAC
hyperparameter grid search is stopped unless an implementation defect is
found.

## Foundation verification

The geometry-only neighborhood artifact is derived from the canonical MPL
`pos_ref` field (`105 x 26 x 3`) and has matrix SHA256
`e08c86c4ae0e3070d0a051f70acd8d602fdeb6cef36f827955a8925fc838c5ac`.
It is not a collection or training artifact.

The read-only V7 Replay audit covers 5,491 rows and reports a Q-rank flip rate
of `0.9821526133673284`; low confidence is enriched in rank-flip rows. The
report is `data/awac/innovation_v1/offline_confidence_audit.json`. This is
diagnostic evidence only; it does not claim causal improvement or method
promotion.

The active confidence plus adaptive-BC-KL synthetic learner path passed its
finite-metric smoke. The disabled innovation path remains the default and its
synthetic update parity test passes; the confidence-only diagnostic path also
matches the frozen Standard AWAC update values. Verification completed with
compileall, 67 focused tests, and `861 passed, 43 skipped, 0 failed` full
pytest. No Unity,
Bridge, collection, Dev100, Final300, or training runtime was executed.
