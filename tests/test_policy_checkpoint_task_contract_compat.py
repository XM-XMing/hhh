"""Policy checkpoint task-contract compatibility tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
from planning.contracts.task import (
    TASK_CONTRACT_ID,
    task_contract_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
AWAC_CHECKPOINT = ROOT / "data" / "awac" / "formal" / "standard_awac_10k_v2" / "checkpoint_last.pt"
BC_CHECKPOINT = ROOT / "data" / "teach" / "2026_6w" / "bc_training" / "checkpoint_best_soft.pt"


def _legacy_awac_task_metadata():
    return {
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_sha256": task_contract_sha256(45),
    }


def test_valid_policy_checkpoint_task_metadata_is_resolved_strictly():
    from planning.contracts.task import resolve_policy_checkpoint_task_contract

    resolved = resolve_policy_checkpoint_task_contract(
        _legacy_awac_task_metadata(), expected_max_primitive_steps=45
    )
    assert resolved["task_contract_schema_version"] == 2
    assert resolved["max_primitive_steps"] == 45
    assert resolved["task_contract_sha256"] == task_contract_sha256(45)


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_contract_id", "unknown_task"),
        ("task_contract_sha256", "0" * 64),
    ],
)
def test_unknown_or_mismatched_policy_task_identity_fails_closed(field, value):
    from planning.contracts.task import resolve_policy_checkpoint_task_contract

    metadata = _legacy_awac_task_metadata()
    metadata[field] = value
    with pytest.raises(ValueError):
        resolve_policy_checkpoint_task_contract(
            metadata, expected_max_primitive_steps=45
        )


def test_explicit_wrong_schema_or_horizon_still_uses_strict_validator():
    from planning.contracts.task import resolve_policy_checkpoint_task_contract

    metadata = _legacy_awac_task_metadata()
    metadata.update(
        {
            "task_contract_schema_version": 2,
            "max_primitive_steps": 40,
        }
    )
    with pytest.raises(ValueError):
        resolve_policy_checkpoint_task_contract(
            metadata, expected_max_primitive_steps=45
        )


def test_real_bc_and_awac_checkpoint_task_metadata_passes():
    torch = pytest.importorskip("torch")
    from planning.contracts.task import resolve_policy_checkpoint_task_contract

    for path in (BC_CHECKPOINT, AWAC_CHECKPOINT):
        if not path.exists():
            pytest.fail("required frozen checkpoint is missing: {}".format(path))
        try:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(str(path), map_location="cpu")
        resolved = resolve_policy_checkpoint_task_contract(
            checkpoint, expected_max_primitive_steps=45
        )
        assert resolved["task_contract_id"] == TASK_CONTRACT_ID
        assert resolved["task_contract_schema_version"] == 2
        assert resolved["max_primitive_steps"] == 45


def test_wrong_observation_contract_is_rejected_by_policy_provenance():
    from planning.evaluation.observation_provenance import (
        resolve_evaluation_observation_provenance,
    )

    metadata = {
        "observation_contract": "legacy_async_telemetry",
        "observation_source": "legacy_async_telemetry",
    }
    with pytest.raises(ValueError):
        resolve_evaluation_observation_provenance(
            metadata,
            expected_observation_contract=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
            runtime_observation_contract=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
            runtime_observation_source=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
            reliable_execution_enabled=True,
            telemetry_observation_enabled=False,
            telemetry_fallback_enabled=False,
            state_depth_exact_endpoint_binding=True,
            allow_override=False,
        )


def test_real_awac_actor_state_loads_after_task_resolution():
    torch = pytest.importorskip("torch")
    from planning.bc.model import build_model
    from planning.contracts.task import resolve_policy_checkpoint_task_contract

    if not AWAC_CHECKPOINT.exists():
        pytest.fail("required frozen checkpoint is missing: {}".format(AWAC_CHECKPOINT))
    try:
        checkpoint = torch.load(
            str(AWAC_CHECKPOINT), map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(str(AWAC_CHECKPOINT), map_location="cpu")
    resolve_policy_checkpoint_task_contract(
        checkpoint, expected_max_primitive_steps=45
    )
    model = build_model(torch.nn, depth_channels=int(checkpoint["depth_history_frames"]))
    model.load_state_dict(checkpoint["actor_state_dict"], strict=True)


def test_real_awac_uses_hash_bound_bc_normalizer_source():
    torch = pytest.importorskip("torch")
    from planning.evaluation.policy_evaluator import (
        resolve_policy_checkpoint_normalizer,
    )

    if not AWAC_CHECKPOINT.exists() or not BC_CHECKPOINT.exists():
        pytest.fail("required frozen checkpoint is missing")
    try:
        checkpoint = torch.load(
            str(AWAC_CHECKPOINT), map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(str(AWAC_CHECKPOINT), map_location="cpu")
    normalizer, source_path = resolve_policy_checkpoint_normalizer(
        checkpoint,
        checkpoint_path=AWAC_CHECKPOINT,
        normalizer_checkpoint_path=BC_CHECKPOINT,
        torch=torch,
        device=torch.device("cpu"),
        expected_max_primitive_steps=45,
    )
    assert source_path == BC_CHECKPOINT.resolve()
    assert normalizer.mean.shape == (22,)
    assert normalizer.std.shape == (22,)


def test_normalizer_source_hash_mismatch_fails_closed(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from planning.evaluation.policy_evaluator import (
        resolve_policy_checkpoint_normalizer,
    )

    if not AWAC_CHECKPOINT.exists():
        pytest.fail("required frozen checkpoint is missing")
    try:
        checkpoint = torch.load(
            str(AWAC_CHECKPOINT), map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(str(AWAC_CHECKPOINT), map_location="cpu")
    wrong_source = tmp_path / "wrong-normalizer.pt"
    wrong_source.write_bytes(b"not-the-recorded-bc-source")
    with pytest.raises(ValueError, match="normalizer source SHA"):
        resolve_policy_checkpoint_normalizer(
            checkpoint,
            checkpoint_path=AWAC_CHECKPOINT,
            normalizer_checkpoint_path=wrong_source,
            torch=torch,
            device=torch.device("cpu"),
            expected_max_primitive_steps=45,
        )


def test_real_awac_runtime_contract_is_resolved_from_training_config():
    torch = pytest.importorskip("torch")
    from planning.evaluation.policy_evaluator import (
        normalize_policy_checkpoint_metadata,
    )

    if not AWAC_CHECKPOINT.exists():
        pytest.fail("required frozen checkpoint is missing")
    try:
        checkpoint = torch.load(
            str(AWAC_CHECKPOINT), map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(str(AWAC_CHECKPOINT), map_location="cpu")
    resolved = normalize_policy_checkpoint_metadata(checkpoint)
    assert resolved["safety_mask"] == "depth"
    assert resolved["execution_mode"] == "continuous"
    assert resolved["policy_runtime_contract_id"] == "policy_runtime_mask_execution"
