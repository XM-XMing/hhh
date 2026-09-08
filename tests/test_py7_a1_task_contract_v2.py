"""PY7-A1 public task-contract seam tests."""

from __future__ import annotations

import copy

import pytest


V1_TASK_SHA = "5862af0f354c408d74cf4904dc7ecf609944553443ae1128097136be979fd30a"


def _bc_manifest():
    from planning.bc.trainer import BC_MMAP_DATASET_CONTRACT_ID
    from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
    from planning.contracts.task import task_contract_fields

    return {
        "contract_id": BC_MMAP_DATASET_CONTRACT_ID,
        "transition_count": 1,
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "reliable_rows": 1,
        "legacy_rows": 0,
        "source_index_sha256": "a" * 64,
        "source_labels_sha256": "b" * 64,
        "source_depth_masks_sha256": "c" * 64,
        "mpl_contract_sha256": "d" * 64,
        "max_steps": 45,
        **task_contract_fields(),
    }


def test_legacy_v1_metadata_and_sha_are_frozen():
    from planning.contracts.task import (
        legacy_task_contract_v1_metadata,
        legacy_task_contract_v1_sha256,
    )

    metadata = legacy_task_contract_v1_metadata()
    assert "task_contract_schema_version" not in metadata
    assert "max_primitive_steps" not in metadata
    assert legacy_task_contract_v1_sha256() == V1_TASK_SHA


def test_v2_metadata_contains_schema_and_max_steps():
    from planning.contracts.task import task_contract_metadata, task_contract_sha256

    metadata = task_contract_metadata()
    assert metadata["task_contract_schema_version"] == 2
    assert metadata["max_primitive_steps"] == 45
    assert task_contract_sha256() != V1_TASK_SHA


def test_v2_same_input_is_deterministic():
    from planning.contracts.task import task_contract_metadata, task_contract_sha256

    first = task_contract_metadata(max_primitive_steps=45)
    second = task_contract_metadata(max_primitive_steps=45)
    assert first == second
    assert task_contract_sha256(max_primitive_steps=45) == task_contract_sha256(
        max_primitive_steps=45
    )


def test_max_steps_change_changes_v2_sha():
    from planning.contracts.task import task_contract_sha256

    assert task_contract_sha256(max_primitive_steps=45) != task_contract_sha256(
        max_primitive_steps=40
    )


def test_v1_and_v2_sha_are_distinct():
    from planning.contracts.task import legacy_task_contract_v1_sha256, task_contract_sha256

    assert legacy_task_contract_v1_sha256() != task_contract_sha256()


def test_formal_v2_artifact_missing_max_steps_is_rejected():
    from planning.bc.trainer import validate_bc_mmap_provenance

    artifact = _bc_manifest()
    artifact.pop("max_primitive_steps")
    with pytest.raises(ValueError, match="max_primitive_steps"):
        validate_bc_mmap_provenance(artifact)


def test_formal_v2_artifact_max_steps_mismatch_is_rejected():
    from planning.bc.trainer import validate_bc_mmap_provenance

    artifact = _bc_manifest()
    artifact["max_primitive_steps"] = 40
    with pytest.raises(ValueError, match="max_primitive_steps"):
        validate_bc_mmap_provenance(artifact)


def test_legacy_v1_artifact_is_rejected_by_formal_v2_validator():
    from planning.bc.trainer import validate_bc_mmap_provenance
    from planning.contracts.task import (
        legacy_task_contract_v1_metadata,
        legacy_task_contract_v1_sha256,
        validate_task_contract,
    )

    artifact = _bc_manifest()
    artifact.pop("task_contract_schema_version")
    artifact.pop("max_primitive_steps")
    artifact.pop("task_contract_mode")
    artifact.update(legacy_task_contract_v1_metadata())
    artifact["task_contract_sha256"] = legacy_task_contract_v1_sha256()
    with pytest.raises(ValueError, match="schema_version"):
        validate_bc_mmap_provenance(artifact)


def test_explicit_historical_mode_validates_v1_and_marks_legacy():
    from planning.contracts.task import (
        legacy_task_contract_v1_metadata,
        legacy_task_contract_v1_sha256,
        validate_legacy_task_contract_v1,
    )

    artifact = copy.deepcopy(legacy_task_contract_v1_metadata())
    artifact["task_contract_sha256"] = legacy_task_contract_v1_sha256()
    validated = validate_legacy_task_contract_v1(artifact)
    assert validated["task_contract_mode"] == "legacy_v1"
