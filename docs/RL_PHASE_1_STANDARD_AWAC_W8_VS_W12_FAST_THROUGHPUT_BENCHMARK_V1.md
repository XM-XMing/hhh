# RL Phase 1 Standard AWAC W8 vs W12 Fast Throughput Benchmark V1

Status: `PASS` for the bounded comparison. The selected formal Collection/AWAC
worker profile is **W8**. W12 completed the same bounded run successfully, but
its committed-transition throughput improvement was only 2.14%, below the
10% promotion gate.

## Scope and common base

This was a runtime throughput/resource comparison only. No Dev100, Final300,
formal 10K run, Actor/Critic hyperparameter change, contract change, or formal
collection was performed.

- Environment: `conda activate xm`, ROS Noetic, `devel` workspace only.
- Runtime capability: direct and subprocess socket probes passed.
- Common resume checkpoint:
  `data/awac/smoke/standard_awac_online_w2_v4/checkpoint_last.pt`
- Common checkpoint SHA256:
  `442e16037ea42ea0c8dabab7bcc19a99a3b648a70a6e834c4657ca3d89d6f953`
- Common starting online progress: 3964 environment steps and 3941 committed
  online transitions.
- Common Replay identity was preserved in independent sparse clones. Source
  Replay metadata SHA256:
  `614d0aa7c109df9e520457cce51736d08918e5edec2322803d310d0cf14d0865`.
- BC checkpoint SHA256:
  `ffa23c9fb1951700e1f876959c51124f9514e1cf691952bebf52c7d34a2aabd2`.
- Mission SHA256:
  `a37d60ad024350825c27c36d984a8403d467717f9b326f2d8c7c079d1b33ec9b`.
- Worker-spec SHA256:
  `75092ab8731833f6ba5a6dbcf41a3b1fcee84cd8056df86c40112b673cd2ae25`.
- MPL contract SHA256:
  `f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d`.
- Task contract SHA256:
  `2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df`.
- Observation contract: `reliable_exact_endpoint_snapshot`.
- Unity player SHA256:
  `61b963943a41f7bfe7b74d8c301032cd3b092756795bb2fcf25be48d8ee91365`.
- Runtime Assembly SHA256:
  `319f9e764cf51078d3b5e439214361f54aa9d5726449e26e4c0a1ad6232f21f0`.
- Bridge SHA256:
  `667c266e96211aabb5a0cd25b8b35b303903d35cbe2d95be8f9d2bbe5e63d789`.

The W8 and W12 resolved configurations differ only in `env_workers`.
All AWAC optimization values, seed 55, device selection, CPU threads,
contracts, mission source, and reliable-v4 runtime settings are identical.

## Resume-budget semantic

The requested v4 checkpoint already contains 3964 Standard online environment
steps. In the current runner, `--online-env-steps` is the cumulative resume
budget. Passing `1000` therefore fails closed before interaction with
`Standard AWAC online progress is outside its budget`.

The first W8 attempt is preserved at:
`data/awac/benchmarks/standard_awac_workers_fast_v1/w8`.

The valid comparison uses the equivalent cumulative budget
`--online-env-steps 4964`, i.e. at most 1000 additional environment steps.
The W12 wave ended at 996 additional steps because of the allowed in-flight
wave shortfall.

## Results

| Metric | W8 | W12 |
|---|---:|---:|
| Workers | 8 | 12 |
| Additional environment steps | 1000 | 996 |
| Additional committed AWAC transitions | 928 | 794 |
| Additional completed episodes | 34 | 29 |
| Wall time, start marker to summary | 121.97 s | 102.17 s |
| Committed transitions/sec | 7.609 | 7.771 |
| Episodes/min | 16.726 | 17.031 |
| Actor updates (increment) | 116 | 99 |
| Critic updates (increment) | 464 | 397 |
| Replay audit | PASS | PASS |
| Checkpoint transaction | COMMITTED | COMMITTED |
| Runtime failure count | 0 | 0 |
| NaN/Inf count | 0 | 0 |

W12 throughput delta relative to W8 is **+2.14%**, not the required +10%.
Both runs have unique runtime identities and unique worker port mappings.

## Resource and learner gates

- W8 observed runtime snapshot: at least 51 GiB RAM available and about
  1.2 GiB swap used; no runtime failure, OOM, port collision, or post-run
  process remained. The first continuous W8 sampler self-matched its monitor
  shell, so W8 peak RSS/process-count fields are explicitly not treated as
  valid peak measurements.
- W12 continuous process-tree sampling recorded minimum available RAM of
  49.75 GiB, maximum aggregate tracked RSS of 8.23 GiB, maximum tracked
  process count of 88, and maximum tracked FD count of 2317. Swap ranged from
  1.178 to 1.216 GiB without sustained growth; disk remained above 141.78 GiB
  free.
- W8 and W12 runtime quality counters were zero, including runtime failure
  count, NaN/Inf count, and non-finite metric count.
- Learner metrics remained finite and bounded. Mean TD loss was 1.432 (W8)
  and 1.465 (W12); mean BC KL was 0.00257 and 0.00217; AWAC weight clipping
  fraction was zero in both; mean twin-Q disagreement was 0.1723 and 0.1726.

## Gate decision

W8 and W12 both pass functional/runtime gates. W12 fails only the promotion
criterion because its committed-transition throughput gain is below 10%.
Therefore:

`FORMAL_SAFE_STANDARD_AWAC_WORKERS = 8`

This decision applies to the bounded Standard AWAC worker profile only. It does
not change Mission, Collection, Label, Mask, or BC DataLoader worker defaults.

## Artifact and immutability record

Valid comparison artifacts are under:
`data/awac/benchmarks/standard_awac_workers_fast_v2/`.

The v4 checkpoint, BC checkpoint, missions, worker spec, Unity, Bridge,
contracts, and formal data were not modified. No commit was created.
