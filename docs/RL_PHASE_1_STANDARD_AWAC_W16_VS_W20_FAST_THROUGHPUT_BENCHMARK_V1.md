# RL Phase 1 Standard AWAC W16 vs W20 Fast Throughput Benchmark V1

Date: 2026-09-05

## Result

The real bounded runtime comparison completed successfully.

RL_PHASE_1_STANDARD_AWAC_W16_VS_W20_FAST_THROUGHPUT_BENCHMARK_V1=PASS
RUNTIME_CAPABILITY=PASS
W16_ELIGIBLE=YES
W20_ELIGIBLE=YES
SELECTED_ENV_WORKERS=16
FORMAL_STANDARD_AWAC_10K_EXECUTED=NO

W16 is selected: 8.700888 committed transitions/sec, 14.36% above the W8
baseline. W20 produced 6.456337 committed transitions/sec, 15.14% below W8
and 25.80% below W16. These benchmark checkpoints are not formal training
starting points.

## Common base and configuration

Both runs used the same v4 Standard AWAC post-warmup checkpoint, seed, mission
source, worker specification, MPL, contracts, device mode, optimizer
configuration, and phase-local budget. The v4 checkpoint contains 3,964 online
environment steps and 3,941 committed online transitions. The effective
resumed budget was 4,964 cumulative environment steps, allowing at most 1,000
additional steps.

| Identity | Value |
| --- | --- |
| checkpoint | data/awac/smoke/standard_awac_online_w2_v4/checkpoint_last.pt |
| checkpoint SHA256 | 442e16037ea42ea0c8dabab7bcc19a99a3b648a70a6e834c4657ca3d89d6f953 |
| starting replay size | 9,083 |
| starting replay metadata SHA256 | 614d0aa7c109df9e520457cce51736d08918e5edec2322803d310d0cf14d0865 |
| seed | 55 |
| budget | 1,000 additional environment steps |
| effective cumulative budget | 4,964 |
| observation contract | reliable_exact_endpoint_snapshot |
| task contract SHA256 | 2c256e920776849a482b05f9478b13bec846b68fefee3dc35febfdfed75bb5df |
| MPL contract SHA256 | f9188067a93dac1cd89020f0e40e250e9a1b35a1c0687587970faeef2be6c99d |

The W16 and W20 resolved configurations were generated from the current
trainer and compared structurally. CONFIG_DIFF_COUNT_EXCLUDING_ENV_WORKERS=0.
The only difference was env_workers:

| Run | env_workers | resolved config SHA256 |
| --- | ---: | --- |
| W16 | 16 | a1fbf5600718c67f049db39911bd10d04ccbc4b57c5e3ae45c338e3c676bcff0 |
| W20 | 20 | 907b874e148d1d481b6c5ff3b3f03bf33f782c182ea3dbd6b4aebc40be64c247 |

Each run used an independent sparse/preallocated replay clone. The source v4
replay was not modified.

## Throughput results

Rates use the phase-local increment and measured runtime-start-to-summary wall
time. W16 consumed 992 environment steps because of the bounded in-flight
wave; W20 consumed the full 1,000.

| Metric | W8 baseline | W16 | W20 |
| --- | ---: | ---: | ---: |
| workers | 8 | 16 | 20 |
| online env steps | 1,000 | 992 | 1,000 |
| committed transitions | 928 | 870 | 638 |
| wall seconds | 121.968388 | 99.989792 | 98.817641 |
| committed transitions/sec | 7.608529 | 8.700888 | 6.456337 |
| env steps/sec | — | 9.921013 | 10.119651 |
| episodes/min | 16.725645 | 19.802021 | 13.965118 |
| actor updates | 116 | 109 | 80 |
| actor updates/sec | 0.951066 | 1.090111 | 0.809572 |
| critic updates | 464 | 435 | 319 |
| critic updates/sec | 3.804264 | 4.350444 | 3.228169 |

W16_VS_W8_THROUGHPUT_DELTA=+14.357037%
W20_VS_W8_THROUGHPUT_DELTA=-15.143424%
W20_VS_W16_THROUGHPUT_DELTA=-25.796804%

## Reliability and learner gates

Both runs returned AWAC_STANDARD_ONLINE=PASS, reached the online budget stop
reason, and passed the strict summary and checkpoint validators.

| Gate | W16 | W20 |
| --- | --- | --- |
| actual worker count | 16 | 20 |
| runtime readiness records | 32 | 40 |
| unique runtime identities | PASS | PASS |
| unique endpoint ports | PASS | PASS |
| reliable-v4 manifest | PASS | PASS |
| Replay Audit | PASS | PASS |
| replay rows | 9,953 | 9,721 |
| BC calibration rows | 5,142 | 5,142 |
| AWAC online rows | 4,811 | 4,579 |
| legacy replay rows | 0 | 0 |
| runtime failures | 0 | 0 |
| transition drops | 0 | 0 |
| NaN/Inf | 0 | 0 |
| summary validator | PASS | PASS |
| checkpoint validator | PASS | PASS |
| checkpoint transaction | COMMITTED | COMMITTED |
| rolling retention | 2 | 2 |
| owned runtime cleanup | PASS | PASS |
| Q divergence | NO | NO |
| Actor divergence | NO | NO |

| Learner metric | W16 | W20 |
| --- | ---: | ---: |
| TD loss mean / P95 | 1.409350 / 3.108484 | 1.352785 / 3.031024 |
| Q mean | 0.787557 | 0.775239 |
| Q std mean | 1.055733 | 1.045732 |
| twin-Q disagreement mean / P95 | 0.171109 / 0.199777 | 0.170709 / 0.197137 |
| AWAC weight mean / max mean | 1.000000 / 2.094776 | 1.000000 / 2.123415 |
| weight clip fraction | 0 | 0 |
| BC KL mean | 0.002432 | 0.001630 |
| Actor entropy mean | 1.725680 | 1.735813 |
| BC/AWAC top-1 disagreement | 0.023833 | 0.020229 |

## Resource evidence

The resource monitor followed each training process tree and excluded the
monitor itself.

| Metric | W16 | W20 |
| --- | ---: | ---: |
| minimum available RAM (GiB) | 48.136 | 46.616 |
| maximum system-used RAM (GiB) | 14.493 | 16.013 |
| maximum tracked RSS (GiB) | 10.315 | 12.332 |
| swap first/last (GiB) | 1.208 / 1.215 | 1.215 / 1.216 |
| within-run swap delta (GiB) | +0.0066 | +0.0010 |
| minimum disk free (GiB) | 141.198 | 141.137 |
| maximum process count | 114 | 142 |
| maximum FD count | 3,064 | 3,808 |
| Unity count at peak | 16 | 20 |
| Bridge count at peak | 16 | 20 |
| ROS master count at peak | 16 | 20 |
| GPU utilization mean | 58.85% | 52.36% |
| GPU utilization P95 | 90.0% | 88.9% |
| GPU memory maximum | 2.318 GB | 2.600 GB |

The CPU sampler was not primed before sampling, so CPU utilization mean and
load average are NOT_CAPTURED_VALIDLY; no CPU claim is made from those fields.
RAM, swap, GPU, process, and cleanup evidence were valid. There was no OOM,
worker failure storm, sustained swap growth, or system responsiveness failure.

## Selection

W16 satisfies the required >=10% improvement over W8 and all reliability and
resource gates. W20 is functionally eligible but provides no throughput
benefit and is substantially slower than W16. The selected profile is W16.
W12 remains historical context only and is not a final candidate in this
comparison.

The formal Standard AWAC start remains:

data/awac/awac_bc60k_formal_critic_calibration_v6/checkpoint_calibration_pass.pt

SHA256:

bb9433d8d9fb2fde2e2148e84a4058c2b328b9d676608d38c00962f821c3d8de

## Scope and verification

The source v4 checkpoint/replay, BC checkpoint, missions, and worker
specification hashes were rechecked unchanged. No algorithm, Unity, Bridge,
Task, Observation, Reward, MPL, or data-layout contract changed. Dev100,
Final300, formal 10K training, and formal collection were not executed.

The machine-readable result is
docs/rl_phase_1_standard_awac_w16_vs_w20_fast_throughput_benchmark_v1.json.
