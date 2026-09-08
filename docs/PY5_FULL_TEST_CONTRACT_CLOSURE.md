# PY5.1 full test contract closure

Date: 2026-08-28

## Scope and baseline

```text
O=/home/xm/XM/xm_ws/src/planning       # read-only historical baseline
P=/home/xm/XM/src                      # only modified tree
CONDA_ENV=xm
UNITY=/home/xm/XM/XMflight             # frozen; UNITY_FINAL_V1=PASS
CXX_FROZEN_FOR_PLANNING=YES
FORMAL_60000_COLLECTION=NOT_RUN
RL_RUN=NOT_RUN
```

PY5.1 closed the test-contract blockers left by the clean PY5 run. The
initial clean P suite was `717 passed, 42 skipped, 39 failed`; the final
clean P suite was `756 passed, 42 skipped` with zero failures. The initial
failure output is retained at `/tmp/xm-py5-1-full-tests.log`.

The failure matrix below is frozen against the initial run. It records the
failure before repair, not a claim that the repaired test still fails.

## Classification summary

```text
INITIAL_TEST_FAILURE_COUNT=39
CLASSIFIED_FAILURE_COUNT=39
UNKNOWN_FAILURE_COUNT=0
STALE_PATH_CONTRACT_COUNT=10
STALE_SOURCE_MANIFEST_COUNT=0
STALE_FIXTURE_ROOT_COUNT=28
STALE_IMPORT_PATH_COUNT=0
STALE_TEXT_ASSERTION_COUNT=1
INTENTIONAL_DEFERRED_SCOPE_COUNT=0
TRUE_RUNTIME_REGRESSION_COUNT=0
TRUE_ALGORITHM_REGRESSION_COUNT=0
```

No failure was classified as a runtime or algorithm regression. No business
assertion was removed, relaxed, skipped, or marked xfail.

## Historical assessment supersession

The older optimized-tree assessment remains historical evidence only:

```text
OPTIMIZED_TREE_ASSESSMENT=HISTORICAL_PRE_MIGRATION_BASELINE
STATUS=SUPERSEDED_BY_CURRENT_GATES
```

The following old conclusions are obsolete under the later C7, M0.1, M0.2,
M0.3, PY0, PY2, PY3, PY4-A, and PY5 evidence:

```text
native Global Route unverified
CUDA/hybrid removal unverified
Bridge split unverified
reliable collection missing
BC mmap provenance missing
BC trainer provenance missing
evaluation provenance missing
relabel/mask provenance missing
```

## Frozen 39-failure matrix

### F01

```text
TEST_FILE=tests/test_endpoint_observation_cache_csharp.py
TEST_NAME=test_unity_endpoint_observation_cache_contract
ASSERTION=compile and execute the endpoint cache C# contract
EXPECTED=final Unity protocol sources and EndpointObservationCache compile; cache harness exits 0
ACTUAL=CS2001; old Assets/Scripts/XMProtocol.cs and EndpointObservationCache.cs were absent
CURRENT_OWNER=Assets/Scripts/Runtime/Observation/EndpointObservationCache.cs plus Runtime/Protocol/*
OLD_OWNER=Assets/Scripts/XMProtocol.cs and Assets/Scripts/EndpointObservationCache.cs
FAILURE_CLASS=STALE_PATH_CONTRACT
REQUIRED_ACTION=compile the final unique owners and retain the cache behavior assertions
```

### F02

```text
TEST_FILE=tests/test_endpoint_observation_capture_ownership_csharp.py
TEST_NAME=test_endpoint_capture_callback_ownership_contract
ASSERTION=compile and run the endpoint capture callback ownership harness
EXPECTED=final EndpointObservationCaptureOwnership compiles and all 6 ownership checks pass
ACTUAL=CS2001; old flat EndpointObservationCaptureOwnership.cs was absent
CURRENT_OWNER=Assets/Scripts/Runtime/Observation/EndpointObservationCaptureOwnership.cs
OLD_OWNER=Assets/Scripts/EndpointObservationCaptureOwnership.cs
FAILURE_CLASS=STALE_PATH_CONTRACT
REQUIRED_ACTION=resolve the final owner and keep the callback ownership checks
```

### F03

```text
TEST_FILE=tests/test_endpoint_observation_snapshot_service_csharp.py
TEST_NAME=test_unity_endpoint_observation_snapshot_service_contract
ASSERTION=compile and run the endpoint snapshot service harness
EXPECTED=final cache, protocol, and snapshot service compile; all 4 service checks pass
ACTUAL=CS2001; old flat cache/service/protocol paths were absent
CURRENT_OWNER=Assets/Scripts/Runtime/Observation/EndpointObservationSnapshotService.cs plus Runtime/Observation/EndpointObservationCache.cs and Runtime/Protocol/*
OLD_OWNER=Assets/Scripts/EndpointObservationSnapshotService.cs and old flat protocol/cache files
FAILURE_CLASS=STALE_PATH_CONTRACT
REQUIRED_ACTION=use final owners and preserve service behavior checks
```

### F04

```text
TEST_FILE=tests/test_endpoint_observation_snapshot_wire_codec_csharp.py
TEST_NAME=test_unity_endpoint_observation_snapshot_wire_codec_contract
ASSERTION=compile and run the endpoint snapshot wire codec harness
EXPECTED=final wire codec and protocol core compile and the codec contract exits 0
ACTUAL=CS2001; old flat XMProtocol.cs and snapshot wire codec paths were absent
CURRENT_OWNER=Assets/Scripts/Runtime/Protocol/EndpointObservationSnapshotWireCodec.cs plus Runtime/Protocol/ProtocolCanonical.cs, ProtocolModels.cs, and ProtocolConstants.cs
OLD_OWNER=Assets/Scripts/XMProtocol.cs and Assets/Scripts/EndpointObservationSnapshotWireCodec.cs
FAILURE_CLASS=STALE_PATH_CONTRACT
REQUIRED_ACTION=compile final protocol owners and retain wire assertions
```

### F05

```text
TEST_FILE=tests/test_observation_snapshot_v4_csharp_contract.py
TEST_NAME=test_csharp_observation_snapshot_codec_matches_golden_vectors
ASSERTION=compile C# snapshot codec and compare the v4 golden vectors
EXPECTED=final protocol and identity owners compile; golden vectors match exactly
ACTUAL=CS2001; old flat XMProtocol.cs was absent
CURRENT_OWNER=Assets/Scripts/Runtime/Protocol/* plus Assets/Scripts/Runtime/Observation/ObservationSnapshotIdentityRegistry.cs
OLD_OWNER=Assets/Scripts/XMProtocol.cs
FAILURE_CLASS=STALE_PATH_CONTRACT
REQUIRED_ACTION=resolve final protocol/identity owners and keep golden-vector equality
```

### F06

```text
TEST_FILE=tests/test_primitive_execution_command_admission_csharp.py
TEST_NAME=test_csharp_command_admission_contract
ASSERTION=compile and run the command admission harness
EXPECTED=final command admission and protocol owners compile; all 5 admission checks pass
ACTUAL=CS2001; old flat command admission/protocol paths were absent
CURRENT_OWNER=Assets/Scripts/Runtime/Primitive/PrimitiveExecutionCommandAdmission.cs plus Runtime/Protocol/*
OLD_OWNER=Assets/Scripts/PrimitiveExecutionCommandAdmission.cs and Assets/Scripts/XMProtocol.cs
FAILURE_CLASS=STALE_PATH_CONTRACT
REQUIRED_ACTION=use final owners and retain all admission checks
```

### F07

```text
TEST_FILE=tests/test_primitive_execution_command_wire_csharp.py
TEST_NAME=test_csharp_decodes_python_command_wire
ASSERTION=decode the Python command wire fixture in C#
EXPECTED=final command admission, wire codec, and protocol owners compile and decode the same fixture
ACTUAL=CS2001; old flat command/protocol paths were absent
CURRENT_OWNER=Assets/Scripts/Runtime/Primitive/PrimitiveExecutionCommandWireCodec.cs and PrimitiveExecutionCommandAdmission.cs plus Runtime/Protocol/*
OLD_OWNER=Assets/Scripts/PrimitiveExecutionCommandWireCodec.cs, PrimitiveExecutionCommandAdmission.cs, and XMProtocol.cs
FAILURE_CLASS=STALE_PATH_CONTRACT
REQUIRED_ACTION=resolve final owners and preserve cross-language wire equality
```

### F08

```text
TEST_FILE=tests/test_primitive_execution_v4_lifecycle_csharp.py
TEST_NAME=test_csharp_v4_result_lifecycle_contract
ASSERTION=compile and run the result lifecycle harness
EXPECTED=final lifecycle/protocol owners compile and all 18 lifecycle checks pass
ACTUAL=CS2001; old flat result lifecycle/protocol paths were absent
CURRENT_OWNER=Assets/Scripts/Runtime/Primitive/PrimitiveExecutionResultLifecycle.cs plus Runtime/Protocol/*
OLD_OWNER=Assets/Scripts/PrimitiveExecutionResultLifecycle.cs and Assets/Scripts/XMProtocol.cs
FAILURE_CLASS=STALE_PATH_CONTRACT
REQUIRED_ACTION=resolve the final owner; retain result, ACK, commit, and failure semantics
```

### F09

```text
TEST_FILE=tests/test_primitive_execution_v4_reliable_transport_csharp.py
TEST_NAME=test_csharp_v4_reliable_result_transport_contract
ASSERTION=compile and run reliable result transport checks
EXPECTED=final lifecycle, adapter, receiver, and protocol owners compile; all 7 transport checks pass
ACTUAL=CS2001; old flat lifecycle/adapter/receiver/protocol paths were absent
CURRENT_OWNER=Assets/Scripts/Runtime/Primitive/PrimitiveExecutionResultLifecycle.cs, PrimitiveExecutionResultTransportAdapter.cs, ReliableResultReceiver.cs plus Runtime/Protocol/*
OLD_OWNER=Assets/Scripts/PrimitiveExecutionResultLifecycle.cs, PrimitiveExecutionResultTransportAdapter.cs, ReliableResultReceiver.cs, and XMProtocol.cs
FAILURE_CLASS=STALE_PATH_CONTRACT
REQUIRED_ACTION=resolve final owners and retain reliable transport behavior checks
```

### F10

```text
TEST_FILE=tests/test_primitive_execution_v4_runtime_integration_csharp.py
TEST_NAME=test_csharp_runtime_event_integration_contract
ASSERTION=compile and run runtime event, result, and telemetry-drop checks
EXPECTED=final controller/session/integration/result/protocol owners compile and all 23 integration checks pass
ACTUAL=CS2001; old flat runtime integration/result/protocol paths were absent; the old harness API also did not match the frozen final integration API
CURRENT_OWNER=Assets/Scripts/Runtime/Primitive/PrimitiveExecutionRuntimeIntegration.cs, PrimitiveExecutionController.cs, PrimitiveExecutionSession.cs, PrimitiveExecutionResultLifecycle.cs, and PrimitiveExecutionResultTransportAdapter.cs
OLD_OWNER=Assets/Scripts/PrimitiveExecutionRuntimeIntegration.cs, result files, and XMProtocol.cs
FAILURE_CLASS=STALE_PATH_CONTRACT
REQUIRED_ACTION=use final owners and a test-only harness adapter while retaining every lifecycle/result/telemetry assertion
```

### F11

```text
TEST_FILE=tests/test_p0_m2_single_worker_runner.py
TEST_NAME=test_real_runner_success_always_writes_completed_summary
ASSERTION=fake real runtime supports the current run_real constructor and completed-summary path
EXPECTED=FakeRealRuntime exposes the DirectRuntimePortProfile seam and summary is written
ACTUAL=AttributeError: FakeRealRuntime has no attribute port_profile
CURRENT_OWNER=python/planning/diagnostics/reliable_single_worker.py::_RealRuntime.port_profile
OLD_OWNER=tests/test_p0_m2_single_worker_runner.py::FakeRealRuntime without port_profile
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=add the current port_profile field to the test fixture; keep summary and cleanup assertions
```

### F12

```text
TEST_FILE=tests/test_p0_m2_single_worker_runner.py
TEST_NAME=test_real_runner_exposes_command_accounting_metrics
ASSERTION=fake real runtime reaches command-accounting summary output
EXPECTED=the current runtime fixture has port profile and command metrics are emitted
ACTUAL=AttributeError: FakeRealRuntime has no attribute port_profile
CURRENT_OWNER=python/planning/diagnostics/reliable_single_worker.py::_RealRuntime.port_profile
OLD_OWNER=tests/test_p0_m2_single_worker_runner.py::FakeRealRuntime without port_profile
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=update only the fixture constructor field and retain metric assertions
```

### F13

```text
TEST_FILE=tests/test_p0_m2_single_worker_runner.py
TEST_NAME=test_real_runner_execution_failure_writes_failed_summary
ASSERTION=fake runtime execution failure is persisted as a failed summary
EXPECTED=the current runtime fixture can be constructed and failure cleanup/summary semantics remain observable
ACTUAL=AttributeError: FakeRealRuntime has no attribute port_profile
CURRENT_OWNER=python/planning/diagnostics/reliable_single_worker.py::_RealRuntime.port_profile
OLD_OWNER=tests/test_p0_m2_single_worker_runner.py::FakeRealRuntime without port_profile
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=add port_profile to the fixture; retain fail-closed failure accounting
```

### F14

```text
TEST_FILE=tests/test_p0_m2_single_worker_runner.py
TEST_NAME=test_real_runner_final_accounting_error_writes_partial_summary
ASSERTION=final accounting error produces a partial summary and cleanup
EXPECTED=the current runtime fixture reaches final accounting and preserves partial-summary semantics
ACTUAL=AttributeError: FakeRealRuntime has no attribute port_profile
CURRENT_OWNER=python/planning/diagnostics/reliable_single_worker.py::_RealRuntime.port_profile
OLD_OWNER=tests/test_p0_m2_single_worker_runner.py::FakeRealRuntime without port_profile
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=add port_profile to the fixture; retain partial-result and cleanup assertions
```

### F15

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_dry_run_resolves_disjoint_manifest_without_unity
ASSERTION=guarded SAC dry-run resolves the disjoint manifest without launching Unity
EXPECTED=DRY_RUN=PASS with P source/devel/install resolution and no Unity launch
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh workspace setup resolver
OLD_OWNER=the test's stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=allow current devel/install/source checkout roots while preserving dry-run and no-Unity assertions
```

### F16

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_resolves_independent_training_and_full_eval_cadence
ASSERTION=training and full-evaluation cadence resolve independently
EXPECTED=the cadence contract is reached after workspace bootstrap
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh workspace setup resolver
OLD_OWNER=the test's stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=fix only setup-root resolution; preserve cadence values and assertions
```

### F17

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_awac_guarded_wrapper_dry_run_uses_shared_runner_without_unity
ASSERTION=AWAC wrapper uses the shared guarded runner and does not launch Unity in dry-run
EXPECTED=shared-runner dry-run contract output
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_awac_guarded.sh -> scripts/run_sac_guarded.sh
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=resolve current workspace setup through the shared runner; do not alter AWAC/SAC math
```

### F18

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_keeps_non_aborted_checkpoint_on_normal_resume_path
ASSERTION=normal resume keeps a non-aborted checkpoint
EXPECTED=resume contract reaches the checkpoint guard and preserves the checkpoint
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh workspace setup resolver and checkpoint guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=repair bootstrap path only; retain checkpoint/resume semantics
```

### F19

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_baseline_mismatch_blocks_collection_before_runtime[summary_delta0-success]
ASSERTION=success baseline mismatch blocks collection before runtime
EXPECTED=BC_BASELINE_SANITY=RED and collection is not opened
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh baseline guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=reach the unchanged baseline guard after current setup resolution
```

### F20

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_baseline_mismatch_blocks_collection_before_runtime[summary_delta1-collision]
ASSERTION=collision baseline mismatch blocks collection before runtime
EXPECTED=BC_BASELINE_SANITY=RED and collection is not opened
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh baseline guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=reach the unchanged baseline guard after current setup resolution
```

### F21

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_baseline_mismatch_blocks_collection_before_runtime[summary_delta2-dead_end]
ASSERTION=dead-end baseline mismatch blocks collection before runtime
EXPECTED=BC_BASELINE_SANITY=RED and collection is not opened
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh baseline guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=reach the unchanged baseline guard after current setup resolution
```

### F22

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_baseline_mismatch_blocks_collection_before_runtime[summary_delta3-timeout]
ASSERTION=timeout baseline mismatch blocks collection before runtime
EXPECTED=BC_BASELINE_SANITY=RED and collection is not opened
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh baseline guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=reach the unchanged baseline guard after current setup resolution
```

### F23

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_baseline_mismatch_blocks_collection_before_runtime[summary_delta4-far]
ASSERTION=far baseline mismatch blocks collection before runtime
EXPECTED=BC_BASELINE_SANITY=RED and collection is not opened
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh baseline guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=reach the unchanged baseline guard after current setup resolution
```

### F24

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_baseline_mismatch_blocks_collection_before_runtime[summary_delta5-altitude]
ASSERTION=altitude baseline mismatch blocks collection before runtime
EXPECTED=BC_BASELINE_SANITY=RED and collection is not opened
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh baseline guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=reach the unchanged baseline guard after current setup resolution
```

### F25

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_baseline_evaluator_failure_blocks_collection[PROTOCOL_ERROR]
ASSERTION=baseline evaluator protocol error blocks collection
EXPECTED=BC_BASELINE_SANITY=RED and collection is not opened
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh baseline evaluator guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=repair setup-root resolution and preserve evaluator-failure blocking
```

### F26

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_baseline_evaluator_failure_blocks_collection[evaluator_failure]
ASSERTION=baseline evaluator failure blocks collection
EXPECTED=BC_BASELINE_SANITY=RED and collection is not opened
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh baseline evaluator guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=repair setup-root resolution and preserve evaluator-failure blocking
```

### F27

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_opens_collection_only_after_exact_baseline_sanity
ASSERTION=collection opens only after exact baseline sanity
EXPECTED=BC_BASELINE_SANITY=GREEN before collection is opened
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh baseline gate and collection handoff
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=repair setup-root resolution; preserve exact-green gate ordering
```

### F28

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_awac_guarded_runner_refuses_sac_output_directory
ASSERTION=AWAC runner refuses a SAC output directory
EXPECTED=protected output-directory contract returns the expected refusal status
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist; expected refusal status was not reached
CURRENT_OWNER=scripts/run_awac_guarded.sh and shared run_sac_guarded.sh output guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=repair bootstrap path only; preserve output ownership guard
```

### F29

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_rejects_protected_extra_argument[--total-env-steps 7---total-env-steps]
ASSERTION=protected total-env-steps override is rejected
EXPECTED=protected-argument refusal status and marker
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh protected argument guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=repair bootstrap path only; retain protected-argument guard
```

### F30

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_rejects_protected_extra_argument[--env-workers 2---env-workers]
ASSERTION=protected env-workers override is rejected
EXPECTED=protected-argument refusal status and marker
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh protected argument guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=repair bootstrap path only; retain protected-argument guard
```

### F31

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_rejects_protected_extra_argument[--candidate-interaction-rate 0.01---candidate-interaction-rate]
ASSERTION=protected candidate interaction-rate override is rejected
EXPECTED=protected-argument refusal status and marker
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh protected argument guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=repair bootstrap path only; retain protected-argument guard
```

### F32

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_rejects_protected_extra_argument[--actor-trust-batch-size 256---actor-trust-batch-size]
ASSERTION=protected actor-trust-batch-size override is rejected
EXPECTED=protected-argument refusal status and marker
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh protected argument guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=repair bootstrap path only; retain protected-argument guard
```

### F33

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_rejects_protected_extra_argument[--algorithm awac---algorithm]
ASSERTION=protected algorithm override is rejected
EXPECTED=protected-argument refusal status and marker
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh protected argument guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=repair bootstrap path only; retain algorithm-selection guard
```

### F34

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_refuses_state_that_requires_intervention
ASSERTION=intervention-required state is refused
EXPECTED=intervention refusal status and no unsafe continuation
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh intervention/state guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=repair bootstrap path only; preserve fail-closed state handling
```

### F35

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_skips_unity_for_identical_deployable_policy
ASSERTION=identical deployable policy skips Unity and keeps the guarded path
EXPECTED=zero-status no-Unity path with identity-bound policy handling
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh deployable-policy identity guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=repair bootstrap path only; preserve no-Unity optimization and identity checks
```

### F36

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_recovers_only_hash_bound_identical_policy_false_block
ASSERTION=false block is recoverable only for a hash-bound identical policy
EXPECTED=identity-bound recovery path and unchanged guard decision
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist
CURRENT_OWNER=scripts/run_sac_guarded.sh policy hash/recovery guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=repair bootstrap path only; preserve hash-bound recovery semantics
```

### F37

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_requires_two_catastrophic_results_to_block[False]
ASSERTION=one catastrophic result does not cross the two-result block threshold
EXPECTED=the parameterized false case returns status 3 under the unchanged threshold contract
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist; observed status 1
CURRENT_OWNER=scripts/run_sac_guarded.sh catastrophic-result counter/guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=repair bootstrap path only; preserve two-catastrophic-result threshold
```

### F38

```text
TEST_FILE=tests/test_sac_guarded_runner_contract.py
TEST_NAME=test_guarded_runner_requires_two_catastrophic_results_to_block[True]
ASSERTION=two catastrophic results cross the block threshold
EXPECTED=the parameterized true case returns status 0 after the unchanged threshold behavior
ACTUAL=source failure: /home/xm/devel/setup.bash did not exist; observed status 1
CURRENT_OWNER=scripts/run_sac_guarded.sh catastrophic-result counter/guard
OLD_OWNER=the stale /home/xm/devel/setup.bash fixture root
FAILURE_CLASS=STALE_FIXTURE_ROOT
REQUIRED_ACTION=repair bootstrap path only; preserve two-catastrophic-result threshold
```

### F39

```text
TEST_FILE=tests/test_script_contracts.py
TEST_NAME=test_unit_contract_script[test_contract_naming.py]
ASSERTION=versioned business names are either absent or explicitly justified
EXPECTED=bounded allowlist, every entry has symbol/path, reason, and removal phase; unexplained count is zero
ACTUAL=unbounded regex reported active protocol, provenance, evaluation, and frozen AWAC/SAC compatibility names as prohibited
CURRENT_OWNER=tests/contracts/test_contract_naming.py explicit 36-entry allowlist
OLD_OWNER=the global regex assertion over every _v1/_v2/_v4 text match
FAILURE_CLASS=STALE_TEXT_ASSERTION
REQUIRED_ACTION=replace global regex exemption with bounded allowlist; defer AWAC/SAC cleanup until new BC is evaluated
```

## Repairs and protected behavior

The repairs were limited to test owners, fixtures, one proven production
workspace-path seam, and contract documentation:

```text
UNITY_PATH_CONTRACT_TESTS=PASS
FAKE_RUNTIME_CONTRACT_TESTS=PASS
AWAC_SAC_RUNNER_CONTRACT_TESTS=PASS
VERSION_NAMING_TEST=PASS
UNEXPLAINED_VERSIONED_BUSINESS_NAME_COUNT=0
AWAC_TO_SAC_IMPORT_COUNT=3
ALGORITHM_CHANGED=NO
RUNTIME_BEHAVIOR_CHANGED=NO
```

The Unity tests now resolve actual final owner paths, assert class/struct/
enum/interface presence, assert old duplicate owners are absent, and assert
the owner is unique. The runtime integration test uses a test-only harness
adapter for the frozen final API; it still checks result lifecycle, ACK/
commit, endpoint snapshot, and telemetry-drop behavior. The lifecycle
fixture's standalone generic failure uses `INTERNAL_ERROR`; a `COLLISION`
failure remains required to carry endpoint observation in the final v4
protocol.

The only production file changed was `scripts/run_sac_guarded.sh`. Its
bootstrap now accepts existing devel/install setup files and an intentionally
unbuilt source checkout, while preserving the guarded runner's cadence,
checkpoint, resume, baseline, selection, and fail-closed semantics. No
algorithm file was modified.

The naming allowlist has 36 exact path/symbol entries. Active protocol and
artifact identities are retained; AWAC/SAC compatibility names are explicitly
`DEFERRED_UNTIL_NEW_BC_EVALUATED`. There is no global regex exemption,
directory exclusion, skip, or xfail.

## Before/after test gates

```text
SKIP_COUNT_BEFORE=42
SKIP_COUNT_AFTER=42
XFAIL_COUNT_BEFORE=0
XFAIL_COUNT_AFTER=0

PY5_INITIAL_FULL_SUITE=717 passed, 42 skipped, 39 failed
PY5_FINAL_FULL_SUITE=756 passed, 42 skipped
FINAL_TEST_FAILURE_COUNT=0
FULL_PYTHON_TESTS=PASS
CANONICAL_CLI=11/11
PY5_COMPILEALL=PASS
```

The 42 skips are unchanged environment/scope gates: Unity/ROS runtime,
transport, and unavailable cross-mission gates. No skip or xfail was added.

## Required post-production gates

The production path-resolver change was followed by a fresh P clean
devel/install build at `/tmp/xm-py5-1-devel-final` and
`/tmp/xm-py5-1-install-final`; both passed. The C++ warning remained the
pre-existing unused `AsInt32` warning. No formal MPL was generated into P.

The fresh staged evidence is under
`/tmp/xm-py5-1-staged-e2e-final/`:

```text
STAGED_PRE_BC_E2E=PASS
2-worker collection=8 attempted, 4 accepted, 150 reliable rows
relabel=4 episodes, 114 transitions, RESULT=PASS
depth_masks=114 transitions, RESULT=PASS
dataset_audit=4/4 episodes, provenance_checked=True, RESULT=PASS
bc_mmap=4 episodes, 114 transitions, RESULT=PASS
bc_training=1 epoch, train=85, val=29, quality_pass=True
reliable_evaluation=1 audit-only episode, RESULT=PASS
legacy_rows=0
telemetry_lookup_count=0
snapshot_missing_count=0
state_depth_skew_max_ns=0
frame_contract_failures=0
```

The 12-worker infrastructure smoke used the existing 150-mission audited
fixture, isolated non-overlapping port bases, `MAX_EPISODES=12`,
`MAX_STEPS=3`, and `TARGET_ACCEPTED=0`. All 12 workers became ready and
finished with unique runtime identities and ports; no collector errors or
reliability violations occurred. Its parent exit status was nonzero only
because the intentionally zero-accepted run failed the separate
`accepted_nonzero` quality gate.

```text
STAGED_TWELVE_WORKER_SMOKE=PASS
WORKER_SPEC_COUNT=12
WORKER_ID_CONTIGUOUS=True
RUNTIME_IDS_UNIQUE=True
PORT_UNIQUE=True
ALL_WORKERS_COMPLETE=True
ALL_RELIABLE_ROWS_POSITIVE=True
ALL_LEGACY_ZERO=True
ALL_TELEMETRY_LOOKUP_ZERO=True
ALL_SNAPSHOT_MISSING_ZERO=True
ALL_SKEW_ZERO=True
ALL_FRAME_FAILURES_ZERO=True
ALL_COLLECTOR_ERRORS_ZERO=True
ORPHAN_PROCESS_COUNT=0
```

The P-to-O dry-run was rerun with no `--delete`; O data and history remain
outside the sync scope. The O preservation check records zero deletions and
zero content/inventory drift for the protected map and history roots.

```text
P_CLEAN_BUILD=PASS
P_INSTALL_BUILD=PASS
STAGED_PRE_BC_E2E=PASS
STAGED_TWELVE_WORKER_SMOKE=PASS
O_PRESERVATION_MANIFEST=PASS
CUTOVER_DRY_RUN=PASS
O_DATA_DELETION_COUNT=0
O_HISTORY_DELETION_COUNT=0
ROLLBACK_PLAN=PASS
```

## Files changed by PY5.1

```text
tests/helpers/__init__.py
tests/helpers/unity_final_paths.py
tests/test_endpoint_observation_cache_csharp.py
tests/test_endpoint_observation_capture_ownership_csharp.py
tests/test_endpoint_observation_snapshot_service_csharp.py
tests/test_endpoint_observation_snapshot_wire_codec_csharp.py
tests/test_observation_snapshot_v4_csharp_contract.py
tests/test_primitive_execution_command_admission_csharp.py
tests/test_primitive_execution_command_wire_csharp.py
tests/test_primitive_execution_v4_lifecycle_csharp.py
tests/test_primitive_execution_v4_reliable_transport_csharp.py
tests/test_primitive_execution_v4_runtime_integration_csharp.py
tests/csharp/PrimitiveExecutionV4LifecycleContract.cs
tests/csharp/PrimitiveExecutionV4RuntimeIntegrationContract.cs
tests/test_p0_m2_single_worker_runner.py
scripts/run_sac_guarded.sh
tests/contracts/test_contract_naming.py
docs/PY5_FULL_TEST_CONTRACT_CLOSURE.md
```

No file under O or Unity was modified. Temporary smoke outputs and failed
fixture-invocation logs are outside both repositories and are retained as
diagnostics; no formal 60,000-row data was collected.

## Final decision

```text
PY5_1_FULL_TEST_CONTRACT_CLOSURE=PASS
P_TO_O_CUTOVER_READY=YES
NEXT_PHASE=PY6 EXECUTE P TO O CUTOVER
```

This is readiness evidence only. PY6 was not executed in PY5.1.
