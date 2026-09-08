# RL Mainline Implementation Consistency Fix and Offline Recertification V1

Status: completed as an offline source/test/diagnostic change. No Unity, Bridge,
ROS runtime, training, Dev100, Final300, or historical artifact rewrite was
performed.

## Evidence boundary

The requested review package was not present on this host:

- `AWAC_CODE_REVIEW_20260906.md`
- `AWAC_CODE_REVIEW_EVIDENCE.zip`
- `reproduce_findings.py`
- `reproduction_results.json`
- `source_excerpts.md`
- `source_manifest.json`

The committed transaction manifests, checkpoints, Replay metadata, existing
TensorBoard files, current source tree, and `方案.md` were available. Therefore
checkpoint identity and the LR03 declared/actual mismatch are artifact-backed;
historical attribution of every review finding remains `UNVERIFIED`.

## Source changes

- `planning.awac.learner`: strict exact-resume optimizer identity; named Critic
  LR handoff overrides that preserve Adam state; hard recovery no longer receives
  adaptive beta scaling; explicit DeltaQ/U metrics and optimizer provenance.
- `planning.awac.trainer`: exact-resume versus calibration-handoff LR modes are
  explicit and verified before the first update.
- `planning.awac.checkpoint`: exact-resume fingerprints include the current
  optional optimizer-LR provenance field while preserving fingerprints for
  historical checkpoints that predate that field.
- `planning.awac.confidence`: v2 mask-aware policy-action contract:
  `a_BC=argmax(masked pi_BC)`, `a_RL=argmax(masked pi_AWAC)`,
  `DeltaQ=Qmin(a_RL)-Qmin(a_BC)`, `U=abs(Q1(a_RL)-Q2(a_RL))`.
- `planning.awac.online_runtime` and `planning.awac.tensorboard`: DeltaQ/U
  fields are retained in cumulative summaries and TensorBoard snapshots.
- `scripts/audit_awac_innovation.py`: production Replay decode and explicit
  metric/causal boundaries; legacy `rank_flip` and unsupported causal wording
  are not emitted by the current diagnostic path.
- `scripts/run_awac_implementation_recertification.py`: read-only fixed-corpus
  CPU/CUDA recertification producer.
- `pyproject.toml` and the contract allowlist: make the existing `scripts.*`
  unit imports and the new diagnostic artifact identities explicit without
  changing production runtime behavior.

## Historical optimizer audit

| Run | Declared Critic LR | Checkpoint actual Critic LR | Actor updates | AWAC / recovery | rejection | Critic updates | env steps | committed rows | Dev100 | conclusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| calibration V7 | 1.0x | 1.0x | 0 | 0 / 0 | 0 | 246 | 6095 | `NOT_PERSISTED` | n/a | valid calibration artifact |
| Standard v2 | 1.0x | 1.0x | 731 | 665 / 66 | 55 | 5144 | 10000 | 9797 | 62/100 | historical result preserved |
| Critic LR03 v1 | 0.3x | **1.0x** | 707 | 646 / 61 | 77 | 5138 | 10000 | 9785 | 61/100 | **not valid evidence for reduced LR** |
| UPS025 v1 | 1.0x | 1.0x | 138 | 138 / 0 | 0 | 2555 | 10000 | 9728 | 65/100 | historical result preserved |
| confidence + adaptive KL v1 retry | 1.0x | 1.0x | 138 | 138 / 0 | 0 | 2555 | 10000 | 9728 | 64/100 | v1 formula identity historical-unverified |

`NOT_PERSISTED` is retained as such; it is not inferred as zero. The LR03
checkpoint and its TensorBoard both show the original Critic groups, so
`LR03_EXPERIMENT_ACTUALLY_APPLIED=NO`. The 61/100 Dev100 result remains a
historical evaluation result, not a reduced-LR conclusion.

## F01-F07 finding matrix

The machine-readable matrix is in:
`data/awac/diagnostics/implementation_recertification_v1_run3/finding_matrix.json`.

- F01: exact resume and explicit named calibration handoff are separated; the
  first-update actual/config LR assertion is enforced.
- F02/F04: production Replay `sample_indices` owns uint8 decode; fixed-corpus
  production/reference inputs and Q/logit/masked-policy outputs matched on CPU
  and CUDA. The unavailable review package prevents historical source replay.
- F03: v2 uses BC/AWAC masked policy actions and the negative-DeltaQ counterexample
  returns zero positive confidence. The closed form is explicitly an
  `IMPLEMENTATION_CHOICE`, not a unique plan formula.
- F05: hard recovery uses `recovery_weight * (bc_kl + trust_tail_penalty)`;
  constant-beta normal, recovery, accepted, and rejected paths match.
- F06/F07: metrics have explicit numerator/denominator, tie/zero-gap handling,
  and `ranking_error_causality=INCONCLUSIVE`. V7 is not treated as ground truth.

## Offline recertification

Fixed corpus: 5491 committed rows from the V7 calibration Replay, read-only,
with the Replay metadata identity recorded in the output. Latest run:

`data/awac/diagnostics/implementation_recertification_v1_run3/`

The same recertification was also run on CPU in `..._run2/`. Both runs report:

- input parity: `PASS`, including single `/255` decode and synthetic 255 -> 1.0;
- model output parity: `PASS` for Q1/Q2, Actor logits, BC logits and masked
  policies;
- CUDA: `PASS` in run3; no mocked GPU result;
- historical actual confidence output is kept separate under
  `historical_actual/`; unavailable old formula identity is marked
  `UNVERIFIED`, never replaced by v2;
- proposed v2 results are under `proposed_formula_offline/` and are not
  historical training metrics or new Dev100 results.

Recertification environment: Python **3.8.20**, NumPy **1.24.4**, PyTorch
**2.4.1**, CUDA runtime **11.8**, NVIDIA **GeForce RTX 4060 Ti**. The CUDA
result is a real available-device result, not a mock.

Required artifacts are present in the latest directory:

`historical_run_validity.json`, `optimizer_actual_lr_audit.json`,
`actor_update_breakdown.json`, `production_offline_parity.json`,
`confidence_contract_v2.json`, `recovery_parity.json`,
`metric_definitions.json`, `finding_matrix.json`,
`historical_actual/`, and `proposed_formula_offline/`.

## Tests and unchanged boundaries

Focused implementation/innovation/trainer/TensorBoard/diagnostic/checkpoint/
script-contract tests: **62 passed, 8 skipped**.
`python -m compileall -q python scripts tests`: **PASS**.
Unfiltered full pytest: **873 passed, 43 skipped, 0 failed**.

Unchanged by this task: BC objective, Task/Observation/Reward/MPL contracts,
Unity, Bridge, Standard AWAC default math, production defaults, Primitive
exploration (still disabled), Replay transaction/storage semantics, and all
historical checkpoint/Replay/summary/TensorBoard files.

The only enabled-innovation semantics changed are the v2 confidence contract
and hard-recovery beta isolation; diagnostic metric semantics changed from
ambiguous rank labels to the explicit definitions above.

Training and evaluation status: `TRAINING_EXECUTED=NO`, `DEV100_EXECUTED=NO`,
`FINAL300_USED=NO`, `PRIMITIVE_EXPLORATION_ENABLED=NO`, `COMMIT=NO`.
